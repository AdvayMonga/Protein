import subprocess
import os
import modal
from modal import App
from modal import Image

app = modal.App()

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==1.5.1")
    .pip_install("biopython>1.78")
    .run_commands("git clone https://github.com/psipred/s4pred")
)

def process_batch(batch):
    results = {}
    for seq_key,seq in batch:
        print(f"processing {seq_key}")
        output = subprocess.check_output(["python","run_model.py", seq], shell=False)
        results[seq_key] = output
    return results

def readfile(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(line.strip().split("\t"))
    return data



def append_output(filename, results):
    with open(filename, "a") as outfile:
        for key, value in results.items():
            outfile.write(f"{key}\t{value}\n")



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
        batches[i] = [start,end]
        start = end
    return batches

def process_allbatches(batches, outfile):
    for batch in batches:
        results = process_batch(batch)
        append_output(outfile, results)

@app.function()
def myprocess(filename, start, end):
    data = readfile(filename)
    mybatch = data[start:end]
    results = process_batch(mybatch)
    outfile = f"outfile_{start}"
    append_output(outfile, results)
    return outfile



    
def combine_files(infiles, outfilename):
    with open(outfilename, "a") as outfile:
        for file in infiles:
            with open(file, "r") as input:
                for line in input:
                    outfile.write(line)
            os.remove(file)


@app.local_entrypoint()
def main(datafile: str, resultsfile: str, num_workers: int):
    data = readfile(datafile)
    batches = create_batches(data, num_workers)
    allpredfiles = []
    for batch in batches:
        predfile = myprocess.local(datafile, batch[0], batch[1])
        allpredfiles.append(predfile)
    combine_files(allpredfiles, resultsfile)

if __name__ == "__main__":
    datafile = "test10.txt"
    resultsfile = "results"
    num_workers = 10
    main(datafile, resultsfile, num_workers)

        

        