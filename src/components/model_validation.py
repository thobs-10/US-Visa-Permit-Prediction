

def evaluate_model(self, exp_key, X_test, y_test)-> str:
        """
        Evaluates a model based on the experiment key and test data.

        Retrieves the experiment from Comet.ml using the provided experiment key,
        and then downloads the best model associated with the experiment.
        The model is then evaluated using the provided test data, and various metrics
        such as accuracy, f1 score, precision, recall, and ROC are calculated.
        These metrics are logged to the experiment in Comet.ml.

        Parameters:
            exp_key (str): The key of the experiment to retrieve from Comet.ml.
            X_test (array-like): The test data to use for model evaluation.
            y_test (array-like): The true labels for the test data.

        Returns:
            tuple: A tuple containing the accuracy, f1 score, precision, recall, and ROC of the model.
        """
        try:
            logging.info(" Starting the evaluation phase")
            # Initialize Comet.ml API
            api = API(api_key=os.getenv('API_KEY'))
        
            api_experiment = api.get_experiment_by_key(exp_key)
            # Get the models with accuracy greater than 90%
            query_condition = (Metric("train_Accuracy") > 0.70)
            matching_api_experiments = api.query(api_experiment.workspace, api_experiment.project_name, query_condition)
            
            if len(matching_api_experiments) > 0:
                # Get the best experiment
                best_experiment = matching_api_experiments[0]
                # Get the tracked model associated with the best experiment
                model_name = best_experiment.get_model_names()
                assets = best_experiment.get_model_asset_list(model_name=model_name[0])
                # Filter out the models
                model_assets = [asset for asset in assets if asset['fileName'].endswith('.pkl')]
                # Download the best model
                if len(model_assets) > 0:
                    model_asset = model_assets[0]
                    model_file_name = model_asset['fileName']
                    model_url = model_asset['curlDownload']
                    # Load the downloaded model
                    os.makedirs('src/models/best_model', exist_ok=True)
                    output_path = f'src/models/best_model/'
                    best_experiment.download_model(model_name[0], output_path= output_path)
                    file_extension="*.pkl"
                    files = glob.glob(os.path.join(output_path, file_extension))
                    latest_file = max(files, key=os.path.getmtime)
                    model = joblib.load(latest_file)
                    # Evaluate the model
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)  
                    f1 = f1_score(y_test, y_pred)  
                    precision = precision_score(y_test, y_pred)  
                    recall = recall_score(y_test, y_pred) 
                    roc_auc = roc_auc_score(y_test, y_pred)  
                    print('-------Evaluation summary---------')
                    print(f"Accuracy: {acc}\nf1 score: {f1}\nPrecision: {precision}\nRecall: {recall}\nRoc: {roc_auc}")
                    # Log metrics for this model
                    best_experiment.log_metric("Accuracy", acc)
                    best_experiment.log_metric("f1", f1)
                    best_experiment.log_metric("Precision", precision)
                    best_experiment.log_metric("Recall", recall)
                    best_experiment.log_metric("ROC", roc_auc)
                    # experiment.log_artifact(model_file_name, f"./{model_file_name}")  # log model artifact to the experiment in Comet.ml
                    logging.info(f"Completed model evaluation.")
                    
                else:
                    print("No models found with accuracy greater than 90%.")
            else:
                print("No experiments found with accuracy greater than 90%.")
            return exp_key
        except Exception as e:
            raise AppException(e, sys)