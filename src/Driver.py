from src.data.RetinaDataset import RetinaDataset
from src.models.RetinaModel import RetinaModel
from src.Experiment import Experiment

exp = Experiment(model_names = ['resnet'],
                data_dir = 'data/processed/{0}_Set/{1}_final/',
                label_file = 'data/processed/{0}_Set/{1}_labels_final.csv',
                results_dir = 'data/processed/results/',
                batch_size = 64,
                num_epochs = 1,
                feature_extract = True,
                learning_rates = [0.02],
                experiment_name = "2.0",
                DEBUG = True)
exp.run(1)