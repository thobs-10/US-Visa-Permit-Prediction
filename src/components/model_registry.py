

def register_model(self, experiment_key):
        try:
            logging.info("Starting the model registration phase")
            api = API(api_key=os.getenv('API_KEY'))
            experiment = api.get_experiment_by_key(experiment_key)
            accuracy_condition = Metric("train_Accuracy") > 0.70
            matching_experiments = api.query(
                experiment.workspace, experiment.project_name, accuracy_condition)
            if matching_experiments:
                best_experiment = matching_experiments[0]
                model_name = best_experiment.get_model_names()[0]
                assets = best_experiment.get_model_asset_list(model_name=model_name)
                feedback = best_experiment.register_model(model_name=model_name)
                if feedback:
                    logging.info("Successfully registered the model")
        except Exception as e:
            raise AppException(e, sys)

