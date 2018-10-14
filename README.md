# Tweet like Trump
Train an LSTM model to generate Trump style tweets.

## Dependencies
* Python 3.5+
* PyTorch 0.4.0

## Usage
### 1. Clone this repo
```bash
$ git clone https://github.com/rliu054/tweet-like-trump
$ cd tweet-like-trump/
```
### 2. Run the model
```bash
$ python main.py --embed_size=50 --hidden_size=50 --epochs=20
```

### 3. Check the generate tweets!
```bash
$ cat output.txt
```


## Acknowledgements
The data used to train this model was grabbed from this [great project](https://github.com/bpb27/trump_tweet_data_archive).

## Author
&copy; [Rui Liu](http://ruiliu.me)
