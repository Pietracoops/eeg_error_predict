from flanker import Participant, FlankerData, load_flanker_data_from_pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from utils import check_class_distribution

from preprocessing import oversample_classes, undersample_classes
import yaml
from gridsearch import gridsearch, randomgridsearch
import pprint
from datetime import datetime
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer

from utils import save_params, load_params, check_pos_int_arg, check_pos_float_arg, save_plot, save_model_stats
from plotting_metrics import (get_f1_score, plot_confusion_matrix_display,
    plot_roc_curve, get_accuracy, get_roc_score, get_precision_recall, get_feature_importance)

class MLAnalysis:
    def __init__(self, paramater_filepath=None):
        self.r_seed = 10
        if paramater_filepath == None:
            self.parameter_filepath = os.getcwd() + "\\parameters.yaml"
        self.params = self.__load_parameters(paramater_filepath)

        # self.models = {'svm': SVC(probability=True),
        #           'random_forest': RandomForestClassifier()}

        self.models = {'random_forest': RandomForestClassifier()}
        # self.model_gs_keys = {'svm': ['svm', 'svm_poly'],
        #                  'random_forest': ['random_forest']}
        self.model_gs_keys = {'svm': ['svm'],
                         'random_forest': ['random_forest']}


    def __load_parameters(self, parameter_path):
        with open(parameter_path, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def prepare_data(self, filepath):
        flankerdata = load_flanker_data_from_pickle(filepath)

        if self.params['no_participant_leakage']:
            self.X_train, self.y_train, self.X_test, self.y_test = flankerdata.split_data()
        else:
            X, y = flankerdata.concatenate_data()
            X_reshaped = X.reshape((X.shape[0], -1))
            self.data = X
            self.labels = y
            check_class_distribution(y)
            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_reshaped, y, test_size=0.2,
                                                                                    random_state=40, stratify=y)

        if self.params['undersampling'] == 1:
            print("Performing under sampling:")
            # Apply sampling strategy
            self.X_train, self.y_train = undersample_classes(self.X_train, self.y_train)
            check_class_distribution(self.y_train)
        elif self.params['oversampling'] == 1:
            if self.params['os_strategy'] == 'SMOTE':
                self.X_train, self.y_train = oversample_classes(self.X_train, self.y_train, strategy="SMOTE",
                                                                ratio=self.params['smote_ratio'])
            elif self.params['os_strategy'] == 'random':
                self.X_train, self.y_train = oversample_classes(self.X_train, self.y_train, strategy="random",
                                                                ratio=self.params['smote_ratio'])

    def grid_search(self, classifier, classifier_key):
        print(f"\nPerforming a random hyperparameter grid search with {self.params['gs_search_iter']}"
              f" iterations and {self.params['gs_num_folds']} folds per iteration. This may take a minute...\n")

        if self.params['gs_random'] == 1:
            estimator, max_value, best_model_name = randomgridsearch(classifier,
                                                    self.X_train,
                                                    self.y_train,
                                                    classifier_key,
                                                    self.r_seed,
                                                    self.params['gs_search_iter'],
                                                    'f1_macro',
                                                    num_folds=self.params['gs_num_folds'])
        else:
            estimator, max_value, best_model_name = gridsearch(classifier,
                                                    self.X_train,
                                                    self.y_train,
                                                    classifier_key,
                                                    'f1_macro',
                                                    num_folds=self.params['gs_num_folds'])

        print(f"\nModel Parameters for {estimator.__class__.__name__}:")
        print("-------------------------------------------------------")
        pprint.pprint(estimator.get_params())
        print("-------------------------------------------------------\n")

        if self.params['gs_save_params']:
            print(f"Saving parameters to params for {estimator.__class__.__name__} model")
            save_params(estimator.get_params(), f"{best_model_name}_{max_value}")

        return estimator, max_value, best_model_name

    def search_model_domain(self):
        best_models = []
        best_scores = []
        best_gs_keys = []
        for model_name, model in self.models.items():
            print(f"Searching {model_name}...")
            best_model, best_score, best_gs_key = self.grid_search(model, self.model_gs_keys[model_name])
            best_models.append(best_model)
            best_scores.append(best_score)
            best_gs_keys.append(best_gs_key)
            print(f"{model_name} - {best_gs_key}: Best Score: {best_score}")

        # Get the results on the held out 10% of data
        for i in range(len(best_models)):
            model = best_models[i]
            print(f"Starting Validation Set for {best_gs_keys[i]}")
            test_preds = model.predict(self.X_test)
            test_probs = model.predict_proba(self.X_test)

            f1 = get_f1_score(self.y_test, test_preds)
            print(f"[{best_gs_keys[i]}]: F1 Macro Score: {f1}")

            acc = get_accuracy(self.y_test, test_preds)
            print(f"[{best_gs_keys[i]}]: Accuracy: {acc}")

            roc_auc = get_roc_score(self.y_test, test_probs)
            print(f"[{best_gs_keys[i]}]: Area under ROC curve: {roc_auc}")

            precision, recall = get_precision_recall(self.y_test, test_preds)
            print(f"[{best_gs_keys[i]}]: Precision: {precision}")
            print(f"[{best_gs_keys[i]}]: Recall: {recall}")

            conf_mat = plot_confusion_matrix_display(self.y_test, test_preds)
            roc_curve = plot_roc_curve(self.y_test, test_probs)

            if self.params['save_plots']:
                # Get the date and time
                now = datetime.now()
                now_string = now.strftime("%d_%m_%y_%H_%M_%S")
                save_plot(conf_mat, f"{best_gs_keys[i]}_{f1}_conf_mat_{now_string}.png")
                save_plot(roc_curve, f"{best_gs_keys[i]}_{f1}_roc_curve_{now_string}.png")


    def launch_model(self, model, model_name, model_params):

        print("Fitting model...")
        model.fit(self.X_train, self.y_train)

        print("Done Fitting...")
        if self.params['cross_validation']:
            # Perform cross-validation
            cv_preds = cross_val_predict(model, self.X_train, self.y_train, cv=self.params['cv_count'])

            # Calculate metrics on the training set (cross-validation predictions)
            cv_f1 = get_f1_score(self.y_train, cv_preds)
            print(f"[{model_name}]: Training F1 Macro Score (CV): {cv_f1}")

            cv_acc = get_accuracy(self.y_train, cv_preds)
            print(f"[{model_name}]: Training Accuracy (CV): {cv_acc}")

        test_preds = model.predict(self.X_test)
        test_probs = model.predict_proba(self.X_test)

        f1 = get_f1_score(self.y_test, test_preds)
        print(f"[{model_name}]: F1 Macro Score: {f1}")

        acc = get_accuracy(self.y_test, test_preds)
        print(f"[{model_name}]: Accuracy: {acc}")

        roc_auc = get_roc_score(self.y_test, test_probs)
        print(f"[{model_name}]: Area under ROC curve: {roc_auc}")

        precision, recall = get_precision_recall(self.y_test, test_preds)
        print(f"[{model_name}]: Precision: {precision}")
        print(f"[{model_name}]: Recall: {recall}")

        conf_mat = plot_confusion_matrix_display(self.y_test, test_preds)
        roc_curve = plot_roc_curve(self.y_test, test_probs)

        results_dict = {'f1': f1,
                        'accuracy': acc,
                        'roc_auc': roc_auc,
                        'precision': precision,
                        'recall': recall}

        save_model_stats(f"{model_name}_{f1}", model_params, results_dict, conf_mat, roc_curve)





