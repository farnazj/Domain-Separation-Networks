To construct representations of text that are generalizable to another domain but that also capture the unique characteristics of their specific domain, we train three encoders: a private source encoder, a private target encoder, and a shared encoder for both the source and the target domains. The representations of the shared and the private encoder for each domain are nudged towards orthogonality. A shared decoder is also added to the architecture for increased generalizability and for ensuring that the private encoders avoid trivial solutions. The decoder attempts to reconstruct the original embedding given a hidden representation of the source or the target domain.

To train the model, run the following command:
python main.py --train --model_name lstm/cnn --save_path PATH_TO_DIRECTORY  

To test the model, run the following command:
python main.py --test --model_name lstm/cnn --target_encoder target_encoder.pt --shared_encoder shared_encoder.pt

