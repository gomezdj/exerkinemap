import json
# Assuming 'tu' (ToolUniverse CLI/Agent) is already initialized and authenticated
# import tu 

def collect_cross_organ_spatial_data():
    """
    Searches and retrieves spatially-registered HuBMAP samples for 
    cross-organ spatial omics analysis (e.g., CODEX, Xenium, spatial RNA-seq).
    """
    # Define target organs using 2-letter RUI codes
    # Left Kidney (LK), Spleen (SP), Heart (HT), Left Lung (LL), Liver (LV) Spleen (SP), Pancreas (PA), Skin (SK), Brain (BR), Bone Marrow (BM), Small Intestine (SI), Large Intestine (LI), Uterus (UT), Placenta (PL), Bladder (BL), Lymph Node (LY)
    target_organs = ['LK', 'SP', 'HT', 'LL', 'LV','SP','PA','SK','BR','BM','SI','LI','UT','PL','BL','LY'] 
    
    # Dictionary to hold our mapped data
    cross_organ_spatial_map = {}

    for organ in target_organs:
        print(f"--- Querying spatially-registered samples for Organ: {organ} ---")
        
        # 1. Search for spatially registered tissue sections
        search_query = {
            "name": "HuBMAP_search_samples",
            "arguments": {
                "organ": organ,
                "sample_category": "section", # Sections are typically used for spatial assays
                "registered_only": True,      # Crucial for CCF/RUI coordinates
                "limit": 5                    # Adjust limit as needed
            }
        }
        
        try:
            # Execute search
            search_result = tu.run(search_query)
            
            # Handle string JSON vs list depending on how your tu.run() parses output
            samples = search_result if isinstance(search_result, list) else json.loads(search_result)
            
            organ_spatial_records = []
            
            for sample in samples:
                # Assuming the tool returns a list of dictionaries with a 'hubmap_id' key
                hubmap_id = sample.get('hubmap_id')
                
                if hubmap_id:
                    print(f"  Retrieving full CCF/RUI spatial record for {hubmap_id}...")
                    
                    # 2. Retrieve exact 3D dimensions and reference organ placement
                    get_sample_query = {
                        "name": "HuBMAP_get_sample",
                        "arguments": {
                            "hubmap_id": hubmap_id
                        }
                    }
                    
                    sample_details = tu.run(get_sample_query)
                    organ_spatial_records.append({
                        "hubmap_id": hubmap_id,
                        "spatial_metadata": sample_details
                    })
            
            # Add to main collection
            cross_organ_spatial_map[organ] = organ_spatial_records

        except Exception as e:
            print(f"Error querying organ {organ}: {e}")

    return cross_organ_spatial_map

def retrieve_sample_spatial_record(hubmap_id: str):
    """
    Retrieves the full record for a single HuBMAP tissue Sample by its HuBMAP ID.
    Extracts key CCF/RUI spatial registration details for downstream analysis.
    """
    print(f"Fetching full spatial record for HuBMAP ID: {hubmap_id}...\n")
    
    # 1. Define the query using the tool specification
    query = {
        "name": "HuBMAP_get_sample",
        "arguments": {
            "hubmap_id": hubmap_id
        }
    }
    
    try:
        # 2. Execute the tool
        result = tu.run(query)
        
        # Handle string JSON vs dict depending on tu.run() output format
        sample_record = json.loads(result) if isinstance(result, str) else result
        
        # 3. Parse and display the key spatial and metadata attributes
        organ = sample_record.get('organ', 'Unknown')
        category = sample_record.get('sample_category', 'Unknown')
        owning_group = sample_record.get('group_name', 'Unknown')
        parent_donor_id = sample_record.get('donor', {}).get('hubmap_id', 'Unknown')
        
        # Spatial specific data
        rui_location = sample_record.get('rui_location')
        ccf_annotations = sample_record.get('ccf_annotations', [])
        
        print(f"--- General Metadata ---")
        print(f"Organ: {organ}")
        print(f"Sample Category: {category}")
        print(f"Owning Group: {owning_group}")
        print(f"Parent Donor ID: {parent_donor_id}")
        
        print(f"\n--- CCF/RUI Spatial Registration ---")
        if rui_location:
            # Assuming rui_location contains placement and dimension data
            placement = rui_location.get('placement_target', 'Unknown reference')
            x_dim = rui_location.get('x_dimension', 'N/A')
            y_dim = rui_location.get('y_dimension', 'N/A')
            z_dim = rui_location.get('z_dimension', 'N/A')
            
            print(f"Placement Target: {placement}")
            print(f"Dimensions (X/Y/Z): {x_dim} / {y_dim} / {z_dim}")
            
            if ccf_annotations:
                print("Overlapping Anatomical Structures (CCF/UBERON):")
                for annotation in ccf_annotations:
                    print(f"  - {annotation}")
            else:
                print("Overlapping Anatomical Structures: None listed")
        else:
            print("Status: No spatial registration (rui_location) found for this sample.")

        return sample_record
        
    except Exception as e:
        print(f"An error occurred while fetching {hubmap_id}: {e}")
        return None

import json
# Assuming 'tu' (ToolUniverse CLI/Agent) is already initialized and authenticated
# import tu 

def search_hubmap_samples(organ=None, sample_category=None, registered_only=False, limit=10):
    """
    Searches HuBMAP biospecimen samples (blocks, sections, organs, or suspensions).
    Allows filtering by organ code, category, and spatial registration status.
    """
    print(f"Executing HuBMAP sample search (Limit: {limit})...")
    
    # 1. Build the arguments dictionary dynamically to only include provided filters
    arguments = {
        "registered_only": registered_only,
        "limit": limit
    }
    
    if organ:
        arguments["organ"] = organ
    if sample_category:
        arguments["sample_category"] = sample_category

    # 2. Construct the tool query
    query = {
        "name": "HuBMAP_search_samples",
        "arguments": arguments
    }
    
    try:
        # 3. Execute the tool
        result = tu.run(query)
        
        # Handle string JSON vs list depending on tu.run() output format
        samples = json.loads(result) if isinstance(result, str) else result
        
        # 4. Display a summary of the results
        print(f"Found {len(samples)} samples matching the criteria.\n")
        
        for i, sample in enumerate(samples, 1):
            hubmap_id = sample.get('hubmap_id', 'Unknown ID')
            cat = sample.get('sample_category', 'Unknown')
            org = sample.get('organ', 'Unknown')
            donor_id = sample.get('donor', {}).get('hubmap_id', 'Unknown Donor')
            
            # The specification notes that spatially-registered samples carry rui_location
            is_registered = "Yes" if sample.get('rui_location') or registered_only else "No"
            
            print(f"{i}. Sample ID: {hubmap_id}")
            print(f"   Organ: {org} | Category: {cat} | Spatially Registered: {is_registered}")
            print(f"   Donor Link: {donor_id}\n")
            
        return samples
        
    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return []

if __name__ == "__main__":
    spatial_data = collect_cross_organ_spatial_data()

    # Example usage using the ID provided in the specification
    target_sample_id = "HBM658.BXNB.873"
        
    full_record = retrieve_sample_spatial_record(target_sample_id)
        
    # Optionally save the raw output for integration into other pipelines
    if full_record:
        output_filename = f"{target_sample_id}_full_record.json"
        with open(output_filename, "w") as f:
            json.dump(full_record, f, indent=4)
        print(f"\nRaw data successfully exported to {output_filename}")


    # Example Usage: Search for spatially-registered target organ sections
    target_organ = "SI"
    category = "section"
        
    search_results = search_hubmap_samples(
        organ=target_organ, 
        sample_category=category, 
        registered_only=True, 
        limit=5
    )
    
    # Export the raw manifest for downstream pipelines or analysis
    if search_results:
        output_filename = f"hubmap_search_{target_organ}_{category}.json"
        with open(output_filename, "w") as f:
            json.dump(search_results, f, indent=4)
        print(f"Full search manifest successfully exported to '{output_filename}'")
    
    # Save the cross-organ mapping data for downstream spatial omics analysis
    with open("cross_organ_spatial_map.json", "w") as f:
        json.dump(spatial_data, f, indent=4)
        
    print("\nData extraction complete. Saved to 'cross_organ_spatial_map.json'")