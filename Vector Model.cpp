#include "stdafx.h" //Precompiled Headers used while working with Visual Studio.
#include <mathimf.h>  //Intel optimized math library. Replace it with math.h if not available.
#include <omp.h> //OpenMP for Parallelization
#include <chrono> //Chrono for Timing
#include <iostream>
//#include <random>
#include <fstream>
#include <string>
//Using Boost Library 
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>



//Simulation Parameters
#define vd 2	//Dimension of Vector. vd>=1
#define ld 2	//Dimension of Lattice. ld>=1
#define n 20	//No of lattice points along each side
#define latlen 400	//Total number of lattice points = n^ld .You'll get segmentation fault if you set it wrong.
#define thetamax 3.14/4.0	//Maximum angle while generating a random angle
#define tempmin 1  //min temperature in units of 0.01
#define tempmax 8  //max temperature in units of 0.01
#define tempsteps 1 //temperature steps in units of 0.01
#define maxthreads 8	//Maximum number of parallel threads to use
#define measurelen 50 //Number of sweeps after which measurement has to be taken
#define measurenum 20000 //Number of measurements to take
#define equibsweep 40000 //Number of sweeps to run to equilibriate
#define folder "output"  //Name of the output directory. Create the folder before running
#define j 1		//coupling parameter in the hamiltonian
double b[vd] = {};	//external field in the hamiltonian

//Some definitions to make timing easier. Change 'steady_clock' to 'monotonic_clock' for older c++ standards
#define startt starttime = chrono::steady_clock::now()  //start counting
#define endt endtime = chrono::steady_clock::now()   //End counting
#define showt chrono::duration_cast<chrono::milliseconds>(endtime - starttime).count()  //show time elapsed

using namespace std;
/*
///Globally defining random number generators
uniform_real_distribution<double> angle(-thetamax, +thetamax);
uniform_real_distribution<double> metro(0, 1);
uniform_int_distribution<unsigned int> vect(0, vd - 1);
uniform_int_distribution<unsigned int> posi(0, n - 1);
mt19937 rng1, rng2;
#pragma omp threadprivate(rng1,rng2,angle,metro,vect,posi,b)
*/

//Mersenne twister Psuedo Random Number Generator with seed = 1 . u means the datatype is long
boost::mt19937 rng1(1u);
//threadprivate makes an independent copy of variable rng1 for each thread so that they dont have to share the same memory
#pragma omp threadprivate(rng1)
//Some random variables we'll eventually use
boost::variate_generator<boost::mt19937&, boost::uniform_int<> > vect(rng1, boost::uniform_int<> (0,vd-1));
boost::variate_generator<boost::mt19937&, boost::uniform_int<> > posi(rng1, boost::uniform_int<>(0, n-1));
boost::variate_generator<boost::mt19937&, boost::uniform_real<> > angle(rng1, boost::uniform_real<>(-thetamax, +thetamax));
boost::uniform_01<boost::mt19937> metro(rng1);
//Making a copy of the random variables for each thread
#pragma omp threadprivate(vect,posi,angle,metro,b)




unsigned int l(unsigned int* pos)
{
	unsigned int len = 0;
	for (unsigned int i = 0;i < ld;i++)
		len += pos[i] * (unsigned int)pow(n, i);
	return len;

}

//Returns vec1.vec2
double dot(double *vec1, double *vec2)
{
	double temp = 0.0;
	for (unsigned int i = 0;i < vd;i++)
		temp += vec1[i] * vec2[i];
	return temp;
}

//vec = {0,0,...,0}
void vectorzero(double* vec)
{
#pragma ivdep
	for (unsigned int i = 0;i < vd;i++)
		vec[i] = 0;
}

//tempvec=vec;
void vectorcpy(double* tempvec, double* vec)
{
#pragma ivdep
	for (unsigned int i = 0; i < vd; i++)
		tempvec[i] = vec[i];
}

//vec1 = vec2 + vec3
void vectoradd(double* vec1, double* vec2, double val)
{
#pragma ivdep
	for (unsigned int i = 0; i < vd; i++)
		vec1[i] += val*vec2[i];
}

//vec1 = val2*vec2 + val3*vec3
void vectoradd(double* vec1, double* vec2, double val2, double* vec3, double val3)
{
#pragma ivdep
	for (unsigned int i = 0; i < vd; i++)
		vec1[i] += val2*vec2[i] + val3*vec3[i];
}

//vec is randomly rotated and the value is stored in tempvec.
//Random rotation is achieved by choosing 2 components of the vector at random and rotating it using 2D rotation matrix.
//It can be shown that this satisfies detailed balance and would eventually span the whole configuration space.
void vectorrandrot(double* tempvec, double* vec)
{
	if(vd==1)
	{
		tempvec[0] = -vec[0];
	}
	else 
	{
		double theta = angle();
		unsigned int t1 = vect(), t2 = vect();

		while (t1 == t2)
		{
			t2 = vect();
		}
#pragma ivdep
		for (unsigned int i = 0;i < vd;i++)
		{
			if (i == t1) 	tempvec[i] = sin(theta)*vec[t2] + cos(theta)*vec[t1];
			else if (i == t2) tempvec[i] = -sin(theta)*vec[t1] + cos(theta)*vec[t2];
			else tempvec[i] = vec[i];
		}
	}
}

//tempvec is assigned with a randomly oriented unit vector. Note: it's not uniformly distribution on a n-sphere.
void vectorrand(double* tempvec)
{
	double abs = 0.0;
	for (unsigned int i = 0;i < vd;i++)
	{
		tempvec[i] = angle();  //Generating a random real.Using angle() since I didn't want to use another random variable.
		abs += tempvec[i] * tempvec[i];  //Calculating absolute value of the vector
	}
	abs = sqrt(abs);
	for (unsigned int i = 0;i < vd;i++)
	{
		tempvec[i] = tempvec[i] / abs; //Normalizing the vector
	}
}

//Returns norm of tempvec
double vectorabs(double* tempvec)
{
	double abs = 0.0;
	for (unsigned int i = 0;i < vd;i++)
	{
		abs += tempvec[i] * tempvec[i];
	}
	return sqrt(abs);
}

void poscpy(unsigned int* temppos, unsigned int* pos)
{
	for (unsigned int i = 0; i < ld; i++)
		temppos[i] = pos[i];
}
void posrand(unsigned int* pos)
{
	for (unsigned int i = 0;i < ld;i++)
	{
		pos[i] = posi();
	}
}

void nearestneighbour(double **lattice, unsigned int pos[ld], double npossum[vd])
{
	unsigned int temppos[ld];
	vectorzero(npossum);
	poscpy(temppos, pos);
	for (unsigned int i = 0;i < ld;i++)
	{
		if (pos[i] == n - 1)
		{
			temppos[i] --;vectoradd(npossum, lattice[l(temppos)], 1.0);
			temppos[i] = 0;vectoradd(npossum, lattice[l(temppos)], 1.0);
		}
		else if (pos[i] == 0)
		{
			temppos[i] ++;vectoradd(npossum, lattice[l(temppos)], 1.0);
			temppos[i] = n - 1;vectoradd(npossum, lattice[l(temppos)], 1.0);
		}
		else
		{
			temppos[i] ++;vectoradd(npossum, lattice[l(temppos)], 1.0);
			temppos[i] -= 2;vectoradd(npossum, lattice[l(temppos)], 1.0);
		}
		temppos[i] = pos[i];
	}
}

void latticeinirand(double** templat)
{
	for (unsigned int i = 0;i < latlen;i++)
	{
		vectorrand(templat[i]);
	}
}
void latticeini1(double** templat)
{
	for (unsigned int i = 0;i < latlen;i++)
	{
		for (unsigned int k = 0;k < vd;k++)
			templat[i][k] = 1.0;
	}
}
double latticeenergy(double **lattice)
{
	double tempenergy = 0.0;
	double npossum[vd] = {};
	unsigned int pos[ld] = {};
	unsigned int kld = 0;
	while (pos[ld - 1] != n)
	{
		kld = 0;
		if (pos[kld] < n)
		{
			vectorzero(npossum);
			nearestneighbour(lattice, pos, npossum);
			tempenergy += -dot(b, lattice[l(pos)]) - j*dot(lattice[l(pos)], npossum);
			pos[kld]++;
		}
		else
			while (pos[kld] == n && pos[ld - 1] != n)
			{
				pos[kld] = 0;
				kld++;
				pos[kld]++;
			}
	}
	return tempenergy / 2.0;
}
void latticemag(double **lattice, double* mag)
{
	vectorzero(mag);
	for (unsigned int i = 0;i < latlen;i++)
		vectoradd(mag, lattice[i], 1.0);
}
double latticemagabs(double **lattice)
{
	double mag[vd] = {};
	for (unsigned int i = 0;i < latlen;i++)
		vectoradd(mag, lattice[i], 1.0);
	return vectorabs(mag);
}
void latticeexport(double** lattice, double t)
{
	ofstream lwrite;
	lwrite.open("./" folder "/lat_ld" + to_string((long long)ld) + "_vd" + to_string((long long)vd) + "_n" + to_string((long long)n) + "_t" + to_string((long double)t) + ".csv");
	for (unsigned int i1 = 0;i1 < latlen;i1++)
	{
		lwrite << lattice[i1][0];
		unsigned int i2 = 1;
		while (i2 < vd)
		{
			lwrite << "," << lattice[i1][i2];
			i2++;
		}
		lwrite << "\n";
	}
	lwrite.close();
}
void latticecopy(double** newlattice, double** oldlattice)
{
	for (unsigned int i = 0;i < latlen;i++)
		for (unsigned int k = 0;k < vd;k++)
			newlattice[i][k] = oldlattice[i][k];
}


double runmcmc(double **lattice, unsigned int sweeps, double t, int measure)
{
	unsigned int pos[ld],temppos[ld];
	double acceptance = 0.0;
	double tempvec[vd], ediff, energy, mag[vd];
	double theta;
	unsigned int t1 , t2 ;
	double* vec;

	//Initializing energy and magnetization
	ofstream fileenergy;
	ofstream filemag;
	if (measure > 1)
	{
		energy = latticeenergy(lattice);
		latticemag(lattice, mag);

		fileenergy.open("./" folder "/latenergy_ld" + to_string((long long)ld) + "_vd" + to_string((long long)vd) + "_n" + to_string((long long)n) + "_t" + to_string((long double)t) + ".csv");
		filemag.open("./" folder "/latmag_ld" + to_string((long long)ld) + "_vd" + to_string((long long)vd) + "_n" + to_string((long long)n) + "_t" + to_string((long double)t) + ".csv");
	}

	for (unsigned int i = 0;i < sweeps;i++)
	{
		for (unsigned int k = 0;k < latlen;k++)
		{
			posrand(pos);
			//nearestneighbour(lattice, pos, npossum);
			vec = lattice[l(pos)];

			//Start:: Nearest Neighbour Code
			//vectorzero(npossum);
			double npossum[vd] = {};
			poscpy(temppos, pos);
			for (unsigned int i2 = 0;i2 < ld;i2++)
			{
				if (pos[i2] == n - 1)
				{
					temppos[i2] --;vectoradd(npossum, lattice[l(temppos)], 1.0);
					temppos[i2] = 0;vectoradd(npossum, lattice[l(temppos)], 1.0);
				}
				else if (pos[i2] == 0)
				{
					temppos[i2] ++;vectoradd(npossum, lattice[l(temppos)], 1.0);
					temppos[i2] = n - 1;vectoradd(npossum, lattice[l(temppos)], 1.0);
				}
				else
				{
					temppos[i2] ++;vectoradd(npossum, lattice[l(temppos)], 1.0);
					temppos[i2] -= 2;vectoradd(npossum, lattice[l(temppos)], 1.0);
				}
				temppos[i2] = pos[i2];
			}
			//End:: Nearest Neighbour Code

			//vectorrandrot(tempvec, vec);
			
			//Start:: Vector random rotation code
			theta = angle();
			t1 = vect(); t2 = vect();

			while (t1 == t2)
			{
				t2 = vect();
			}
#pragma ivdep
			for (unsigned int i1 = 0;i1 < vd;i1++)
			{
				if (i1 == t1) 	tempvec[i1] = sin(theta)*vec[t2] + cos(theta)*vec[t1];
				else if (i1 == t2) tempvec[i1] = -sin(theta)*vec[t1] + cos(theta)*vec[t2];
				else tempvec[i1] = vec[i1];
			}
			//End:: Vector random rotation code
			
			ediff = - j*(dot(npossum, tempvec) - dot(npossum, vec));

			if (metro() < exp(-ediff / t))
			{
				if (measure > 1)
				{
					energy += ediff;
					vectoradd(mag, vec, -1.0, tempvec, +1.0);
				}
				vectorcpy(vec, tempvec);
				acceptance += 1.0;
			}
		}

		if (measure > 1 && i%measurelen==0)
		{
			fileenergy << energy << "\n";
			filemag << mag[0];
			for (unsigned int i = 1;i < vd;i++)
				filemag << "," << mag[i];
			filemag << "\n";
		}
		
	}
		filemag.close();
		fileenergy.close();
	return acceptance / (sweeps*latlen);
}



int main()
{
	omp_set_num_threads(maxthreads);
	#pragma omp parallel 
	{
		auto starttime = chrono::steady_clock::now();  //Change 'steady_clock' to 'monotonic_clock' for older c++ standards
		auto endtime = chrono::steady_clock::now();  //Change 'steady_clock' to 'monotonic_clock' for older c++ standards
		double acc;
		//Dynamically allocating array to store the lattice (each thread has its own separate copy)
		double **lattice = new double*[latlen];
		for (unsigned int i = 0; i < latlen; ++i)
		{
			lattice[i] = new double[vd];
		}
		#pragma omp for  
		for (int t = tempmin;t <= tempmax; t+= tempsteps)
		{
			startt;
			latticeinirand(lattice);
			//run simulation for 40000 sweeps
			acc=runmcmc(lattice, equibsweep, 0.01*(double)t, 0);
			endt;
			printf("Equilibration:: t=%f__acc=%f__Time=%I64d__Thread=%d\n", 0.01*(double)t, acc, showt, omp_get_thread_num());

			latticeexport(lattice, 0.01*(double)t);
			
			startt;
			acc = runmcmc(lattice, measurelen*measurenum, 0.01*(double)t, 2);
			endt;
			printf("Measurement:: t=%f__acc=%f__Time=%I64d__Thread=%d\n", 0.01*(double)t, acc, showt, omp_get_thread_num());
			
		}
	//Deleting dynamically allocated array storing the lattice
		for (unsigned int i = 0; i < latlen; ++i)
		{
			delete[] lattice[i];
		}
		delete[] lattice;
	}
	cout << "The End";

	//getchar();
	return;
}
