#domain adaptation
To train the model, run the following command:
python main.py --train --model_name lstm/cnn --save_path model_name.pt 

To test the model, run the following command:
python main.py --test --model_name lstm/cnn --snapshot model_name.pt

Download embeddings from http://nlp.stanford.edu/data/glove.840B.300d.zip
