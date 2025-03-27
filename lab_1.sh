python3 -m venv pyenv

#Activate the virtual enviroment
source pyenv/bin/activate

#install the required packages
pip3 install pandas numpy matplotlib

rebuild() {
    rm .depend

    #clean the directory
    make clean

    #compile the code
    make
}

#Run the code
rebuild && python3 compilers.py --mode headless
mv benchmarks/stats/* labs/lab1/compilers

rebuild && python3 flags.py --mode headless
mv benchmarks/stats/* labs/lab1/flags