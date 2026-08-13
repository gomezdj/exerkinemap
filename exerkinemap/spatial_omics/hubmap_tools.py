## HuBMAP_get_dataset (Type: HuBMAPTool)#
import json
# Assuming 'tu' (ToolUniverse CLI/Agent) is already initialized and authenticated
# import tu 

class HuBMAPTools:
    """
    A unified integration module for querying and retrieving HuBMAP 
    dataset metadata, provenance, and organ availability.
    """

    @staticmethod
    def get_dataset(hubmap_id: str):
        """
        Get detailed metadata for a specific HuBMAP dataset by its HuBMAP ID.
        """
        print(f"Fetching metadata for dataset: {hubmap_id}...")
        query = {
            "name": "HuBMAP_get_dataset",
            "arguments": {
                "hubmap_id": hubmap_id
            }
        }
        
        try:
            result = tu.run(query)
            # Parse output depending on how tu.run() returns data
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"Error fetching dataset {hubmap_id}: {e}")
            return None

    @staticmethod
    def get_dataset_provenance(uuid_or_id: str):
        """
        Retrieve a HuBMAP dataset’s biological provenance lineage.
        Answers: 'which donor, organ, and tissue blocks/sections produced this dataset?'
        """
        print(f"Tracing provenance for: {uuid_or_id}...")
        query = {
            "name": "HuBMAP_get_dataset_provenance",
            "arguments": {
                "uuid": uuid_or_id
            }
        }
        
        try:
            result = tu.run(query)
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"Error tracing provenance for {uuid_or_id}: {e}")
            return None

    @staticmethod
    def list_organs():
        """
        List all 43 organs available in the HuBMAP atlas.
        Returns organ names, RUI codes, UBERON ontology IDs, and CUI codes.
        """
        print("Retrieving HuBMAP organ dictionary...")
        query = {
            "name": "HuBMAP_list_organs",
            "arguments": {}
        }
        
        try:
            result = tu.run(query)
            return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            print(f"Error retrieving organ list: {e}")
            return None

    @staticmethod
    def search_datasets(organ=None, dataset_type=None, text_query=None, status="Published", limit=10):
        """
        Search HuBMAP published datasets by organ, assay type, or free text.
        """
        print(f"Executing HuBMAP dataset search (Limit: {limit})...")
        
        arguments = {
            "status": status,
            "limit": limit
        }
        
        # Dynamically append optional filters
        if organ:
            arguments["organ"] = organ
        if dataset_type:
            arguments["dataset_type"] = dataset_type
        if text_query:
            arguments["query"] = text_query
            
        query_struct = {
            "name": "HuBMAP_search_datasets",
            "arguments": arguments
        }
        
        try:
            result = tu.run(query_struct)
            datasets = json.loads(result) if isinstance(result, str) else result
            print(f"Found {len(datasets)} datasets matching the search criteria.")
            return datasets
        except Exception as e:
            print(f"Error executing dataset search: {e}")
            return []

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Example HuBMAP Implementation Pipeline
    # ---------------------------------------------------------
    
    # 1. Check available organ codes 
    # organs = HuBMAPTools.list_organs()
    
    # 2. Search for Left Kidney (LK) CODEX datasets
    target_organ = "SI"
    assay_type = "CODEX"
    
    search_results = HuBMAPTools.search_datasets(
        organ=target_organ, 
        dataset_type=assay_type, 
        limit=2
    )
    
    # 3. Iterate through search results to pull deeper metadata and provenance
    if search_results:
        for idx, dataset in enumerate(search_results, 1):
            hubmap_id = dataset.get('hubmap_id')
            
            print(f"\n--- Processing Result {idx}: {hubmap_id} ---")
            
            # Fetch complete metadata record
            metadata = HuBMAPTools.get_dataset(hubmap_id)
            if metadata:
                print(f"  Data Access Level: {metadata.get('data_access_level', 'Unknown')}")
            
            # Fetch the tissue derivation lineage
            provenance = HuBMAPTools.get_dataset_provenance(hubmap_id)
            if provenance:
                print(f"  Lineage Ancestors Found: {len(provenance)}")