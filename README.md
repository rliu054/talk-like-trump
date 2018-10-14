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

## Sample output
Here are some samples from the generates tweets. We can see some iconic phrases the POTUS love to use.

```
* please mr trump please run for potus thank you <EOS> 
* thank you virginia and i will be on on monday at 900 pm <EOS> 
* im of paul ryan is a gorgeous guy <EOS> 
* you should be president you would make my vote right save america <EOS> 
* if obamacare doesnt have the answers <EOS> 
* keep calling my big plan for the american worker ratings right because i had the course right now also true <EOS> 
```


## Acknowledgements
The data used to train this model was grabbed from this [great project](https://github.com/bpb27/trump_tweet_data_archive).

## Author
&copy; [Rui Liu](http://ruiliu.me)
