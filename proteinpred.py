# This program takes in arbitrary protein sequences and computes Psipred (S4pred) secondary structure predictions
# It uses modal and multiple workers to compute the predictions in parellel
#
# Run this program like this: 
# modal run preteinpred.py --num-workers (the number of workers) --results file (name of file) --datafile (name of datafile)
#

import subprocess
import os
import random
import modal
from modal import Image

app = modal.App()

# The image that is used for processing remotely.
process_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .apt_install("wget")
    .pip_install("torch==1.13.1")
    .pip_install("biopython>1.78")
    .pip_install("modal")
    .run_commands("mkdir diffusebio")
    .workdir("/diffusebio")
    .run_commands("git clone https://github.com/psipred/s4pred")
    .workdir("/diffusebio/s4pred")
    .run_commands("wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz")
    .run_commands("tar -xvzf weights.tar.gz")
    .copy_local_file("proteinpred.py")
)

# Runs S4pred on one batch and writes out the output to a given directory
# where there is one file for each sequence that is processed.
def process_batch(batch, output_dir):
    seq_path = os.path.join(output_dir,"seqfile")
    with open(seq_path,"a") as seqfile:    
        for seq_key,seq in batch:
           seqfile.write(f">{seq_key}\n{seq}\n")
    subprocess.run(["python","run_model.py", "--save-files", "--silent", "--outdir", output_dir, seq_path], shell=False)
    os.remove(seq_path)

# Returns a dictionary with an entry for each file in the directory.
# The key is the name of the file and the value is the string of all the contents in the file.
def return_gift(dir):
    gift = {}
    dirfiles = os.listdir(dir)
    for file in dirfiles:
        filename = os.path.splitext(os.path.basename(file))[0]
        filepath = os.path.join(dir, file)
        lines = [] 
        with open(filepath, "r") as f:
            for stuff in f:
                lines.append(stuff)
        gift[filename] = "\n".join(lines)
    return gift

# Modal function where one worker runs S4pred on a batch and returns its output as a dictionary. 
# The key is the name of the file and the value is the string of all the contents in the file.          
@app.function(image=process_image)
def myprocess(mybatch):
    outdir = f"outdir_{random.randint(0,1000000)}"
    os.mkdir(outdir)
    process_batch(mybatch,outdir)
    return return_gift(outdir)

# Given a file, reads it and returns the data as a list.
def readfile(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(line.strip().split("\t"))
    return data

# Given the number of workers and the data,
# Takes all the data and divides it into batches of equal size among the workers.
# Returns a list of size num_workers with each entry being the data in a singular batch.
def create_batches(data, num_workers):
    if(len(data) < num_workers):
        num_workers = len(data)
    batch_size = len(data) // num_workers
    rem = batch_size % num_workers
    batches = [None] * num_workers
    start = 0
    for i in range(0, num_workers):
        end = start+batch_size
        if(rem > i):
            end += 1
        batches[i] = data[start:end]
        start = end
    return batches

# Given a list of dictionaries containing the sequence predictions and the name of a file,
# writes out the predictions to the given file.
def write_predictions(indicts, outfilename):
    with open(outfilename, "a") as outfile:
        for dict in indicts:
            for key,pred in dict.items():
                outfile.write(f"\n{key}\n{pred}")

# Main for the modal App.
# Reads the data from the data file,
# Breaks up the data into even batches for each worker,
# Uses modal to parallelize the work,
# Collects the output and writes it out to a results file.
@app.local_entrypoint()
def main(datafile: str, resultsfile: str, num_workers: int):
    data = readfile(datafile)
    batches = create_batches(data, num_workers)
    all_predictions = myprocess.map(batches)
    write_predictions(all_predictions, resultsfile)

# Local Testing
if __name__ == "__main__":
    datafile = "test10.txt"
    resultsfile = "results"
    num_workers = 5
    main(datafile, resultsfile, num_workers)
