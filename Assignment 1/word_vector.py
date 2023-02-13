import gensim.downloader
 
word_vectors_google = gensim.downloader.load('word2vec-google-news-300')
word_vectors_google.save('vectors.bin')