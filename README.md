# cs342-image-processing
1. clone the repo
- cd to where you want to run the code (the file containing the features will be saved here)
- clone the repo by running
git clone https://github.com/r-kidy/cs342-image-processing.git


# 2. setup a virtual environment

python3.11 -m venv cs342-cw
source cs342-cw/bin/activate

# 3. install all requirements (this could take a few mins)

pip3.11 install -r requirements.txt

# 4. run image_processing.py on dcs batch compute 

- if youre running the code on your own computer with a gpu, you dont need to do this. otherwise, this will save about 100h vs running it on the dcs computers.

- for a comprehensive guide, visit: https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute

run the following commands in your terminal (replace [your_uni_id] with your id uxxxxxxx)

ssh -J [your_uni_id]@remote-[last_2_digits_of_id].dcs.warwick.ac.uk [your_uni_id]@kudu-taught.dcs.warwick.ac.uk

cd [where you cloned the repo]

sbatch image_processing.sbatch

# 5. wait about 2-3min for the code to run, it will save the features as "celeba_vit_embeddings.npy"
- in your coursework notebook, access the data with the following python code:

data = np.load("celeba_vit_embeddings.npy")


