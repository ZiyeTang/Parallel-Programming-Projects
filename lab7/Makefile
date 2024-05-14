OBJ=lab7

all:
	nvcc -std=c++11 -I../libwb -rdc=true -c $(OBJ).cu -o $(OBJ).o 
	nvcc -std=c++11 -o $(OBJ) $(OBJ).o ../libwb/lib/libwb.so

clean:
	rm -f $(OBJ).o

cleanall: clean
	rm -f $(OBJ) *.out *.err

run:
	sbatch job.slurm

