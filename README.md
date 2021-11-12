# Transformer
Transformer used in Machine Translation

Torchtext is used to get access to dataset Multi30k. I also used torchtext to do preprocessing on the dataset and build my data.To know specifically how it works, you can go to data.py. Note that you need to make sure that torchtext==0.6.0.


As there are some difficulties in downloading  "de" and "en" from spaCy, I've downloaded and put de_core_news_sm-3.2.0.tar.gz and en_core_web_sm-3.2.0.tar.gz in the repository. You can use it by "pip install ./de_core_news_sm-3.2.0.tar.gz" and "pip install ./en_core_web_sm-3.2.0.tar.gz'". Note that these data are 3.2.0v, so you need to make sure that your spaCy version==3.2.0. You can also use "python -m spacy en_core_web_sm" to download directly.

For the transformer part, I split all the functions into encoder.py, encoder_layer.py , decoder.py, decoder_layer.py, position.py , SelfAttention.py, NoamOpt.py. For each of these parts, I write specific annotation on how these tensors' sizes changes, which can help you to understand.
Each part of Transformer in the code is the same as the diagram belows.
![5b841d56e1ff41d882e45e09d788724c](https://user-images.githubusercontent.com/51498590/141443833-8e5222b1-68ba-4c20-885b-162f747f7631.png)

The model is finally integrated in seq2seq.py. You can run as ----  python seq2seq.py directly.



