import os
from utils.MCMC import MCMC_Summarizer

def main():
    files = [s for s in os.listdir('results') if '.pkl' in s]
    burnin = [0, 5000, 15000]
    summarizer = MCMC_Summarizer(files, burnin)
    summarizer.summarize()

if __name__ == "__main__":
    main()