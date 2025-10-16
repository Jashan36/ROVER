"""
Batch Processor
Process multiple images offline with progress tracking and result saving
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import json
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

from .real_time_pipeline import RealtimePipeline, PipelineConfig, InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from batch processing"""
    total_images: int
    successful: int
    failed: int
    avg_time_ms: float
    total_time_seconds: float
    output_dir: Path
    summary: Dict


class BatchProcessor:
    """
    Batch processor for offline image analysis
    """
    
    def __init__(
        self,
        pipeline: Optional[RealtimePipeline] = None,
        config: Optional[PipelineConfig] = None,
        output_dir: str = "data/processed",
        num_workers: int = 1,
        save_visualizations: bool = True,
        save_raw_outputs: bool = False
    ):
        """
        Initialize batch processor
        
        Args:
            pipeline: Existing pipeline (or create new)
            config: Pipeline configuration
            output_dir: Directory for outputs
            num_workers: Number of parallel workers (1 = sequential)
            save_visualizations: Save overlay images
            save_raw_outputs: Save numpy arrays
        """
        self.pipeline = pipeline or RealtimePipeline(config)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.save_visualizations = save_visualizations
        self.save_raw_outputs = save_raw_outputs
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_visualizations:
            (self.output_dir / 'overlays').mkdir(exist_ok=True)
            (self.output_dir / 'traversability').mkdir(exist_ok=True)
            (self.output_dir / 'hazards').mkdir(exist_ok=True)
        
        if save_raw_outputs:
            (self.output_dir / 'segmentations').mkdir(exist_ok=True)
            (self.output_dir / 'costmaps').mkdir(exist_ok=True)
        
        logger.info(f"BatchProcessor initialized: {output_dir}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Save visualizations: {save_visualizations}")

    def process_directory(
        self,
        image_dir: Union[str, Path],
        pattern: str = "*.jpg",
        recursive: bool = False,
        max_images: Optional[int] = None
    ) -> BatchResult:
        """
        Process all images in a directory
        
        Args:
            image_dir: Directory containing images
            pattern: File pattern (e.g., "*.jpg", "*.png")
            recursive: Search subdirectories
            max_images: Maximum number of images to process
            
        Returns:
            BatchResult with summary
        """
        image_dir = Path(image_dir)
        
        # Find images
        if recursive:
            image_paths = list(image_dir.rglob(pattern))
        else:
            image_paths = list(image_dir.glob(pattern))
        
        if not image_paths:
            logger.warning(f"No images found in {image_dir} with pattern {pattern}")
            return BatchResult(
                total_images=0,
                successful=0,
                failed=0,
                avg_time_ms=0.0,
                total_time_seconds=0.0,
                output_dir=self.output_dir,
                summary={}
            )
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        return self.process_images(image_paths)

    def process_images(
        self,
        image_paths: List[Path]
    ) -> BatchResult:
        """
        Process list of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            BatchResult with summary
        """
        import time
        start_time = time.time()
        
        results = []
        failed_images = []
        processing_times = []
        
        # Process images
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            if self.num_workers > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {
                        executor.submit(self._process_single, path): path
                        for path in image_paths
                    }
                    
                    for future in as_completed(futures):
                        path = futures[future]
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                                processing_times.append(result.total_time_ms)
                            else:
                                failed_images.append(path)
                        except Exception as e:
                            logger.error(f"Failed to process {path}: {e}")
                            failed_images.append(path)
                        
                        pbar.update(1)
            else:
                # Sequential processing
                for path in image_paths:
                    try:
                        result = self._process_single(path)
                        if result:
                            results.append(result)
                            processing_times.append(result.total_time_ms)
                        else:
                            failed_images.append(path)
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
                        failed_images.append(path)
                    
                    pbar.update(1)
        
        total_time = time.time() - start_time
        
        # Create summary
        summary = self._create_summary(results)
        
        # Save summary
        self._save_summary(summary, results, failed_images)
        
        # Create batch result
        batch_result = BatchResult(
            total_images=len(image_paths),
            successful=len(results),
            failed=len(failed_images),
            avg_time_ms=np.mean(processing_times) if processing_times else 0.0,
            total_time_seconds=total_time,
            output_dir=self.output_dir,
            summary=summary
        )
        
        logger.info(f"\nBatch processing complete:")
        logger.info(f"  Total: {batch_result.total_images}")
        logger.info(f"  Successful: {batch_result.successful}")
        logger.info(f"  Failed: {batch_result.failed}")
        logger.info(f"  Avg time: {batch_result.avg_time_ms:.1f}ms")
        logger.info(f"  Total time: {batch_result.total_time_seconds:.1f}s")
        
        return batch_result

    def _process_single(
        self,
        image_path: Path
    ) -> Optional[InferenceResult]:
        """Process single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load {image_path}")
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            result = self.pipeline.process(image)
            
            # Save outputs
            self._save_outputs(image_path, result, image)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None

    def _save_outputs(
        self,
        image_path: Path,
        result: InferenceResult,
        original_image: np.ndarray
    ):
        """Save all outputs for an image"""
        stem = image_path.stem
        
        # Save visualizations
        if self.save_visualizations:
            if result.overlay is not None:
                overlay_path = self.output_dir / 'overlays' / f"{stem}_overlay.jpg"
                cv2.imwrite(
                    str(overlay_path),
                    cv2.cvtColor(result.overlay, cv2.COLOR_RGB2BGR)
                )
            
            if result.traversability_viz is not None:
                trav_path = self.output_dir / 'traversability' / f"{stem}_traversability.jpg"
                cv2.imwrite(
                    str(trav_path),
                    cv2.cvtColor(result.traversability_viz, cv2.COLOR_RGB2BGR)
                )
            
            if result.hazard_viz is not None:
                hazard_path = self.output_dir / 'hazards' / f"{stem}_hazards.jpg"
                cv2.imwrite(
                    str(hazard_path),
                    cv2.cvtColor(result.hazard_viz, cv2.COLOR_RGB2BGR)
                )
        
        # Save raw outputs
        if self.save_raw_outputs:
            seg_path = self.output_dir / 'segmentations' / f"{stem}_segmentation.npy"
            np.save(seg_path, result.classes)
            
            if result.costmap is not None:
                costmap_path = self.output_dir / 'costmaps' / f"{stem}_costmap.npy"
                np.save(costmap_path, result.costmap)
        
        # Save metadata
        metadata = {
            'image_path': str(image_path),
            'timestamp': result.timestamp,
            'processing_time_ms': result.total_time_ms,
            'best_direction_deg': float(np.degrees(result.best_direction)),
            'direction_scores': result.direction_scores,
            'num_hazards': len(result.hazards),
            'hazard_summary': result.hazard_summary,
            'stats': result.stats
        }
        
        metadata_path = self.output_dir / f"{stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _create_summary(self, results: List[InferenceResult]) -> Dict:
        """Create summary statistics from results"""
        if not results:
            return {}
        
        # Aggregate statistics
        processing_times = [r.total_time_ms for r in results]
        avg_traversability = [r.stats['avg_traversability'] for r in results]
        num_hazards = [len(r.hazards) for r in results]
        
        # Terrain distribution
        all_terrain_dist = {}
        for r in results:
            for terrain, ratio in r.stats['terrain_distribution'].items():
                if terrain not in all_terrain_dist:
                    all_terrain_dist[terrain] = []
                all_terrain_dist[terrain].append(ratio)
        
        avg_terrain_dist = {
            terrain: float(np.mean(ratios))
            for terrain, ratios in all_terrain_dist.items()
        }
        
        # Hazard types
        all_hazard_types = {}
        for r in results:
            for htype, count in r.hazard_summary.get('by_type', {}).items():
                all_hazard_types[htype] = all_hazard_types.get(htype, 0) + count
        
        summary = {
            'num_images': len(results),
            'performance': {
                'avg_time_ms': float(np.mean(processing_times)),
                'min_time_ms': float(np.min(processing_times)),
                'max_time_ms': float(np.max(processing_times)),
                'std_time_ms': float(np.std(processing_times)),
                'avg_fps': float(1000.0 / np.mean(processing_times))
            },
            'terrain': {
                'avg_distribution': avg_terrain_dist,
                'avg_traversability': float(np.mean(avg_traversability)),
                'min_traversability': float(np.min(avg_traversability)),
                'max_traversability': float(np.max(avg_traversability))
            },
            'hazards': {
                'total_detected': int(np.sum(num_hazards)),
                'avg_per_image': float(np.mean(num_hazards)),
                'max_per_image': int(np.max(num_hazards)),
                'by_type': all_hazard_types
            }
        }
        
        return summary

    def _save_summary(
        self,
        summary: Dict,
        results: List[InferenceResult],
        failed_images: List[Path]
    ):
        """Save summary to file"""
        summary_path = self.output_dir / 'batch_summary.json'
        
        full_summary = {
            'summary': summary,
            'failed_images': [str(p) for p in failed_images]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(full_summary, f, indent=2)
        
        logger.info(f"Summary saved: {summary_path}")


# Convenience function
def process_image_directory(
    image_dir: str,
    output_dir: str = "data/processed",
    model_path: Optional[str] = None,
    pattern: str = "*.jpg",
    num_workers: int = 1,
    max_images: Optional[int] = None
) -> BatchResult:
    """
    Quick function to process directory of images
    
    Args:
        image_dir: Directory containing images
        output_dir: Output directory
        model_path: Path to model weights
        pattern: File pattern
        num_workers: Number of parallel workers
        max_images: Maximum images to process
        
    Returns:
        BatchResult
    """
    from .realtime_pipeline import create_pipeline
    
    pipeline = create_pipeline(model_path=model_path)
    
    processor = BatchProcessor(
        pipeline=pipeline,
        output_dir=output_dir,
        num_workers=num_workers
    )
    
    return processor.process_directory(
        image_dir=image_dir,
        pattern=pattern,
        max_images=max_images
    )


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Batch Processor Test")
    print("=" * 60)
    
    # Create test images
    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating 5 test images in {test_dir}...")
    for i in range(5):
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(test_dir / f"test_{i:03d}.jpg"), test_img)
    
    # Process
    print("\nProcessing images...")
    result = process_image_directory(
        image_dir=str(test_dir),
        output_dir="data/processed_test",
        num_workers=2
    )
    
    print("\nBatch Result:")
    print(f"  Total: {result.total_images}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Avg Time: {result.avg_time_ms:.1f}ms")
    print(f"  Total Time: {result.total_time_seconds:.1f}s")
    
    if result.summary:
        print("\nSummary:")
        print(f"  Avg Traversability: {result.summary['terrain']['avg_traversability']:.3f}")
        print(f"  Total Hazards: {result.summary['hazards']['total_detected']}")
