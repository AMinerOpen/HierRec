# HierRec

Explore hierarchical relation from raw text.

# Dependencies

* Python 3

# Steps

* First, you have raw text corpus.

Using [AWOE](https://github.com/AMinerOpen/AWOE) to extract keywords from documents. And generate a json line file similar to [this file](http://lfs.aminer.cn/awoe/data.zip), where each line contains a json with key words of a document and some other information. Unzip and put it in the *data/* folder.

* Second, find potential hierarchy with keyword jl file.

run *main.py* to get result. (Before this, you should download the corresponding model as suggested in AWOE, or just move the content of this repo into your cloned AWOE.)

There are some parameters you can tune for your own data in this step. See *src/config.py*.

If you can provide a stopword dictionary, things will get better. In our scenario, we use [XLore API](https://xloreapi.docs.apiary.io/#) to search the word, if no entity with same name shows up or the type of entity is "字词", the word will be put into stopword dictionary. The stopword dictionary is also in the *data/* folder in the downloaded file above.


