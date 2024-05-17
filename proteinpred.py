import subprocess
import os
import modal
from modal import App
from modal import Image

app = modal.App()

process_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .apt_install("wget")
    .pip_install("torch==1.5.1")
    .pip_install("biopython>1.78")
    .pip_install("modal")
    .run_commands("git clone https://github.com/psipred/s4pred")
    .run_commands("cd s4pred")
    .run_commands("wget https://raw.githubusercontent.com/AdvayMonga/Protein/main/proteinpred.py")
)

def process_batch(batch, output_dir):
    seq_path = os.path.join(output_dir,"seqfile")
    with open(seq_path,"a") as seqfile:    
        for seq_key,seq in batch:
           print(f"processing {seq_key}")
           seqfile.write(f">{seq_key}\n{seq}\n")
    subprocess.run(["python","run_model.py", "--save-files", "--silent", "--outdir", output_dir, seq_path], shell=False)
    os.remove(seq_path)

def readfile(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(line.strip().split("\t"))
    return data




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

@app.function(image=process_image)
def myprocess(filename, start, end):
    outdir = f"outdir_{start}"
    os.mkdir(outdir)
    data = readfile(filename)
    mybatch = data[start:end]
    process_batch(mybatch,outdir)
    return outdir

#Output of each batch goes into separate directory
#in myprocess, create the new directory and return the directory location
#the directory is passed into process batch
#combine files -> gets files from each directory and combines them into one

    
def combine_files(infiles, outfilename):
    print(f"infiles: {infiles}, outfilename: {outfilename}")
    with open(outfilename, "a") as outfile:
        for file in infiles:
            filename = os.path.splitext(os.path.basename(file))[0]
            outfile.write(f"\n{filename}\n")
            with open(file, "r") as input:
                for line in input:
                    outfile.write(line)
            os.remove(file)

def combine_directory(indirs, outfilename):
    print(f"combine_director: indirs: {indirs}, outfile: {outfilename}")
    for dir in indirs:
        dirfiles = os.listdir(dir)
        dirpaths = []
        for file in dirfiles:
            dirpath = os.path.join(dir, file)
            dirpaths.append(dirpath)
        combine_files(dirpaths, outfilename)
        os.rmdir(dir)


            


@app.local_entrypoint()
def main(datafile: str, resultsfile: str, num_workers: int):
    data = readfile(datafile)
    batches = create_batches(data, num_workers)
    allpreddirs = []
    for batch in batches:
        preddir = myprocess.local(datafile, batch[0], batch[1])
        allpreddirs.append(preddir)
    combine_directory(allpreddirs, resultsfile)

if __name__ == "__main__":
    datafile = "test10.txt"
    resultsfile = "results"
    num_workers = 5
    main(datafile, resultsfile, num_workers)
