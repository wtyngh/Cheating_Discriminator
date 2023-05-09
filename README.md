# BAC
 BAC (Berkeley Anti-Cheat): detecting cheating in Counter-Strike: Global Offensive



First, download the dataset from: https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset. Upzip all files directly under /BAC.

Second, make sure the sklearn and tensorflow are installed: `pip install -U scikit-learn`; `pip install tensorflow`

Then, run "data_processing.py". This should generate a file "train_test_split.npz" with all the data.

Now, you can run any "train_xxx.py" to train a model, and "evaluate_xxx.py" to evaluate how well the model can detect cheating.

All models have been trained, except linear SVM which returns an error commented in the file, so you can run "evaluate_xxx.py" directly.
