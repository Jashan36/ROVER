def generate_science_notes(region_id, notes_dict):
    return f"Region {region_id}: " + ", ".join([f"{k}={v}" for k,v in notes_dict.items()])
