from utils.prepare_vectordb_from_csv import PrepareVectorDBFromCSV

if __name__ == "__main__":
    from pyprojroot import here
    
    # Specify the path to your CSV file
    csv_file = here("data/csv/cancer.csv")
    
    # Create an instance and run the pipeline
    data_prep = PrepareVectorDBFromCSV(file_path=csv_file)
    data_prep.run_pipeline() 