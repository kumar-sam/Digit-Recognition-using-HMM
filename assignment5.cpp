//assignment 5: Implementation of HMM
//******************************************** loading library and definition
#include "stdafx.h"
#include<stdlib.h>
#include<math.h>
#include<conio.h>
#include<string.h>


#define eval_samples 320    // per frame samples
#define data_path "C:\\Users\\pc\\Desktop\\sp\\Assignment5\\196101005_digitRecognition\\196101005_digitRecognition\\assignment5\\assignment5\\196101005_digit\\"
#define root_path "E:\\IITG\\Coding-assignments-master\\Coding-assignments-master\\PhD semester 1\\Speech processing\\196101005_digitRecognition\\196101005_digitRecognition\\assignment5\\assignment5\\"
#define end_frame_energy_threshold 2.25
#define needed_samples 7040	
#define threshold_frame_set 4	// no. of frames to be observed for setting marker
#define threshold_percentage 10		//threshold for setting marker
#define training_count 20 		// no. of training utterances
#define testing_count 10	// no. of testing utterances
#define digits_count	10	
#define max_frame_count  150		// maximum no. of frames to be considered
#define codebook_size 32
#define msgsize 1024		
#define header "214101048_"
#define N 5				// no. of states in HMM model
#define train_iters 20	// also tells number of times we should re-estimate to get a good HMM model for each obs_seq
#define averaging_count 3	// no. of times averaging done
#define start_energy_thr 2.25
#define path_buffer_size 4096	// buffer size for path
#define sample_rate 16000	// sample rate 
#define stride 80			// sliding amount during overlapping.
#define past_samples 12		// number of past samples in LPC.
#define normalization_value 5000		// normalization factor.
#define to_be_trained 1	// for training

//******************************************** defining globals
long double A[N][N];
long double A_avg[N][N];
long double B[N][codebook_size];
long double B_avg[N][codebook_size];
long double Pi[N];
long double Pi_avg[N];
long double Alpha[max_frame_count][N];
long double Beta[max_frame_count][N];
long double Gamma[max_frame_count][N];
long double Delta[max_frame_count][N];
long double Zeta[max_frame_count-1][N][N]; // xi
int Psi[max_frame_count][N];
int Q_star_t[train_iters][max_frame_count];
long double P_star[train_iters];

long double waveform[100000];
char all_digits[digits_count][2] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
long double tw[past_samples];
long double codebook[codebook_size][past_samples];
int digit_match, overall_match;

//******************************************** functions definition

int read_waveform(){
	FILE *fr;
	long double curr;
	int sample_count = 0;
	int size = sizeof(waveform);
	memset(waveform,0 , sizeof(waveform));
	fr = fopen("normalized.txt", "r");
	if(fr==NULL) printf("read_waveform: can't read.");
	else{
		while(fscanf(fr, "%Lf\n", &curr) != EOF){
			waveform[sample_count] = curr;
			sample_count++;
		}
		fclose(fr);
		return sample_count;
	}
	fclose(fr);
	return -1;
}
long double calc_power(int i, int j){
	long double output = 1;
	int k;
	if(j < 0)
		for(k = j; k < 0; k++)
			output /= i;
	else
		for(k = j; k > 0; k--)
			output *= i;

	return output;
}

long int order_check(long double num){
	long int o = 0;

	while (num < 1 && o > -1000) {
	    num *= 10;
	    o--;
	}

	return o;
}

void load_generated_model(int framecount, int digit)
{
	long double curr_value;
	int i, j;
	FILE *rA, *rB, *rPI;

	char num[3];
	char filenameA[msgsize], filenameB[msgsize], filenamePI[msgsize];

	memset(num,0,3);
	sprintf(num,"%d",digit);

	_snprintf(filenameA,sizeof(filenameA),"%s%s%s", "A_final_model_digit_", num,".txt");
	_snprintf(filenameB,sizeof(filenameB),"%s%s%s", "B_final_model_digit_", num,".txt");
	_snprintf(filenamePI,sizeof(filenamePI),"%s%s%s", "PI_final_model_digit_", num,".txt");
	rA = fopen(filenameA, "r");
	rB = fopen(filenameB, "r");
	rPI = fopen(filenamePI, "r");	

	if(rA)
		for(i = 0; i < N; i++)
			for(j = 0; j < N; j++){
				fscanf(rA, "%Lf", &curr_value);
				A[i][j] = curr_value;
			}

	if(rB)
		for(i = 0; i < N; i++)
			for(j = 0; j < codebook_size; j++){
				fscanf(rB, "%Lf", &curr_value);
				B[i][j] = curr_value;
			}
		
	if(rPI)
		for(i = 0; i < N; i++){
			fscanf(rPI, "%Lf", &curr_value);
			Pi[i] = curr_value;	
		}

	fclose(rA);			
	fclose(rB);			
	fclose(rPI);			


}

void load_initial_model(int framecount, int avg_num, int digit)
{
	long double curr_value;
	int i, j;
	FILE *rA, *rB, *rPI;

	if(avg_num == 0){
		rA = fopen("A.txt", "r");
		rB = fopen("B.txt", "r");
		rPI = fopen("PI.txt", "r");	
	}else{
		char num[3];
		char filenameA[msgsize], filenameB[msgsize], filenamePI[msgsize];

		memset(num,0,3);
		sprintf(num,"%d",digit);

		_snprintf(filenameA,sizeof(filenameA),"%s%s%s", "A_final_model_digit_", num,".txt");
		rA = fopen(filenameA, "r");
		_snprintf(filenameB,sizeof(filenameB),"%s%s%s", "B_final_model_digit_", num,".txt");
		rB = fopen(filenameB, "r");
		_snprintf(filenamePI,sizeof(filenamePI),"%s%s%s", "PI_final_model_digit_", num,".txt");
		rPI = fopen(filenamePI, "r");
	}

	if(rA)
		for(i = 0; i < N; i++)
			for(j = 0; j < N; j++){
				fscanf(rA, "%Lf", &curr_value);
				A[i][j] = curr_value;
			}

	if(rB)
		for(i = 0; i < N; i++)
			for(j = 0; j < codebook_size; j++){
				fscanf(rB, "%Lf", &curr_value);
				B[i][j] = curr_value;
			}


	if(rPI)
		for(i = 0; i < N; i++){
			fscanf(rPI, "%Lf", &curr_value);
			Pi[i] = curr_value;
			
		}
	
	fclose(rA);			
	fclose(rB);			
	fclose(rPI);			
}

int get_B_rowmax(int j){
	int k, mi;
	long double max_b;

	mi = 0;
	max_b = B[j][0];

	for(k = 1; k < codebook_size; k++)
		if(B[j][k] > max_b)
		{
			max_b = B[j][k];
			mi = k;
		}
	return k;
}

//solution problem 1: contains forward and backward solution
long double solnProblemOne(int T, int observations[]){
	//printf("solnProblemOne");
	long double sf_i_j, sb_j_i, pos_model = 0;

	memset(Alpha, 0, N * max_frame_count*sizeof(long double));
	memset(Beta, 0, N * max_frame_count*sizeof(long double));

	// step 1: initialization
	for(int i = 0; i < N; i++){
		Alpha[0][i] = Pi[i] * B[i][observations[0]];// assuming that digits are in the range [0, codebook_size)
		Beta[T-1][i] = 1;
	}
	// step 2: induction
	for(int t = 0; t < T-1; t++)
		for(int j = 0; j < N; j++){
			sf_i_j = 0;

			for(int i = 0; i < N; i++)
				sf_i_j += Alpha[t][i] * A[i][j];

			Alpha[t+1][j] = sf_i_j * B[j][observations[t+1]];
		}
	
	for(int t = T-2; t >= 0; t--)
		for(int i = 0; i < N; i++){
			sb_j_i = 0;
			for(int j = 0; j < N; j++)
				sb_j_i += A[i][j] * B[j][observations[t+1]] * Beta[t+1][j];

			Beta[t][i] = sb_j_i;
		}

	// step 3: termination
	for(int i = 0; i < N; i++)
		pos_model += Alpha[T-1][i];
	if(pos_model < 0)
			printf("\n Error in solnProblemOne - P < 0");

	//printf("\n - P(obs/given model): ");
	return pos_model;	
}

// solution problem 2: calculating state sequence
void solnProblemTwo(int T, int model, int observations[]){
	long double max_delta, delta_result, pstar;
	int i;

	// step 1: initialization 
	memset(Delta, 0, sizeof(long double) * N * max_frame_count);
	memset(Psi, -1, sizeof(int) * N * max_frame_count);	

	for(i = 0; i < N; i++){
		Delta[0][i] = Pi[i] * B[i][observations[0]];	// assuming that digits are in the range [0, codebook_size)	
		Psi[0][i] = -1;	// done again only for keeping the format of the algorithm
	}

	// step 2: recursion
	for(int t = 1; t < T; t++)
		for(int j = 0; j < N; j++){
		// initially, assume that max is at i = 0; accordingly set values of delta and psi - if max encountered later, change accordingly
			i = 0;
			max_delta = Delta[t-1][i] * A[i][j];
			Psi[t][j] = i;
			for(i = 1; i < N; i++){
				delta_result = Delta[t-1][i] * A[i][j];
				if(delta_result > max_delta)
				{
					max_delta = delta_result;
					Psi[t][j] = i;
				}
			}
			Delta[t][j] = max_delta * B[j][observations[t]];
		}

	// step 3: termination
	pstar = Delta[T-1][0];
	Q_star_t[model][T-1] = 0;		// qT * as per the algorithm notation

	for(i = 1; i < N; i++)
		if(Delta[T-1][i] > pstar){
			pstar = Delta[T-1][i];
			Q_star_t[model][T-1] = i;
		}


	// step 4: path backtracking
	for(int t = T-2; t >= 0; t--)
		Q_star_t[model][t] = Psi[t+1][Q_star_t[model][t+1]];

	P_star[model] = pstar;

	//printing purpose -------------------------------------------------
	printf("\n - P* Model: %d ",order_check(pstar));
	printf("\n - State Sequence is: ");
	for(int t=0; t<T; t++){
		printf("%d ", Q_star_t[model][t]);
	}
	if(pstar < 0)
		printf("\n Error sol2(): Pstar value < 0");
	
}

void solnProblemThree(int T, int observations[]){

	long double threshold= calc_power(10, -30);
	memset(Gamma, 0,  N * max_frame_count* sizeof(long double));

	// computing xi values
	for(int t = 0; t < T-1; t++){
		long double temp = 0;
		for(int i = 0; i < N; i++)
			for(int j = 0; j < N; j++){
				Zeta[t][i][j] = Alpha[t][i] * A[i][j] * B[j][observations[t+1]] * Beta[t+1][j];
				temp += Zeta[t][i][j];
			}

		for(int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
				Zeta[t][i][j] /= temp;			

	}

	// computing gamma values
	for(int t = 0; t < T; t++){
		long double temp = 0;
		for(int i = 0; i < N; i++){
			Gamma[t][i] = Alpha[t][i] * Beta[t][i];
			temp += Gamma[t][i];
		}
		for(int i = 0; i < N; i++)
			Gamma[t][i] /= temp;				// final calculation of gamma values for time t and state i
		
	}

	// re-estimation of the model

	// new A
	for(int i = 0; i < N; i++){
		long double temp1 = 0, temp2=0;
		for(int t_hat = 0; t_hat < T-1; t_hat++)
			temp1 += Gamma[t_hat][i];		// denominator computation done

		for(int j = 0; j < N; j++){
			temp2 = 0; 
			for(int t = 0; t < T-1; t++)
				temp2 += Zeta[t][i][j];

			A[i][j] = temp2 / temp1;				// numerator computation for current j value done for outer loop value i
		}
	}


	// new B
	for(int j = 0; j < N; j++)
	{
		long double temp1 = 0, temp2=0;
		for(int t_hat = 0; t_hat < T; t_hat++)
			temp1 += Gamma[t_hat][j];		// denominator computation done		b[j][obs_seqs[seq_num][t+1]]

		int count_b_zeros = 0;
		for(int k = 0; k < codebook_size; k++){
			temp2 = 0;
			for(int t = 0; t < T; t++)
				if(observations[t]==k)
					temp2 += Gamma[t][j];

			B[j][k] = temp2 / temp1;
		}

		int index = get_B_rowmax(j);

		//adding threshold
		for(int k = 0; k < codebook_size; k++){
			if(B[j][k] <= threshold){
				count_b_zeros++;
				B[j][k] += threshold;
			}
		}
		// subtracting value from max
		B[j][index] -= count_b_zeros * threshold;  
	}
	
}

void saveFinalModel(int id){
	char num[3];
	memset(num,0,3);
	sprintf(num,"%d",id);
	char fA[msgsize];
	_snprintf(fA,sizeof(fA),"%s%s%s", "A_final_model_digit_", num,".txt");	
	char fB[msgsize];
	_snprintf(fB,sizeof(fB),"%s%s%s", "B_final_model_digit_", num,".txt");
	char fPI[msgsize];
	_snprintf(fPI,sizeof(fPI),"%s%s%s", "PI_final_model_digit_", num,".txt");
	FILE *wA= fopen(fA, "w");
	FILE *wB= fopen(fB, "w");
	FILE *wPI= fopen(fPI, "w");

	for(int i = 0; i < N; i++)
		fprintf(wPI, "%0.40Lf\t", Pi_avg[i]);

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++)
			fprintf(wA, "%0.40Lf\t", A_avg[i][j]);
		fprintf(wA, "\n");
	}

	for(int i = 0; i < N; i++){
		for(int j = 0; j < codebook_size; j++)
			fprintf(wB, "%0.40Lf\t", B_avg[i][j]);
		fprintf(wB, "\n");
	}

	fclose(wA);
	fclose(wB);
	fclose(wPI);

	printf("\n\nFinal Model files written for digit %d\n\n", id);
}


void init_HMM_params_global(){
	memset(A, 0, sizeof(long double) * N * N);
	memset(B, 0, sizeof(long double) * codebook_size * N);
	memset(Pi, 0, sizeof(Pi));
	memset(A_avg, 0, sizeof(long double) * N * N);
	memset(B_avg, 0, sizeof(long double) * codebook_size * N);
	memset(Pi_avg, 0, sizeof(Pi_avg));
}

void init_HMM_params(){
	memset(A, 0, sizeof(long double) * N * N);
	memset(B, 0, sizeof(long double) * codebook_size * N);
	memset(Pi, 0, sizeof(Pi));
	memset(Alpha, 0, sizeof(long double) * N * max_frame_count);
	memset(Beta, 0, sizeof(long double) * N * max_frame_count);
	memset(Delta, 0, sizeof(long double) * N * max_frame_count);
	memset(Psi, -1, sizeof(int) * N * max_frame_count);
	memset(Gamma, 0, sizeof(long double) * N * max_frame_count);		// clearing array
	memset(Q_star_t, -1, sizeof(int) * max_frame_count * train_iters);	// init. with -1 since 0 is a valid state
	memset(P_star, 0, train_iters);
}

void HMM_model(int T, int observations[]){
	memset(Q_star_t, -1, sizeof(long double) * max_frame_count * train_iters);	// init. with -1 since 0 is a valid state
	memset(P_star, 0, train_iters);

	long double p_prob, c_prob;
	p_prob = solnProblemOne(T, observations);	
	solnProblemTwo(T, 0, observations);	
	solnProblemThree(T, observations);

	for(int model_iter = 1; model_iter < train_iters; model_iter++){
		// checking for improvement in the model re-estimated
		c_prob = solnProblemOne(T, observations);
		if (c_prob > p_prob) //better performance
			p_prob = c_prob;
		else if(c_prob == p_prob) //same performance
			int x=1;
			//printf("\nSuccessive models have same performance!");
		else //reduced performance
			p_prob = c_prob;
	
		// checking quality of improvement using new model to compute new P* value
		solnProblemTwo(T, model_iter, observations);
		solnProblemThree(T, observations);

	}
}

void addIntoAverage(){
	for(int i = 0; i < N; i++) Pi_avg[i] += Pi[i];
	for(int i = 0; i < N; i++) for(int j = 0; j < N; j++) A_avg[i][j] += A[i][j];
	for(int i = 0; i < N; i++) for(int j = 0; j < codebook_size; j++) B_avg[i][j] += B[i][j];
}

void getAverageModel(){
	for(int i = 0; i < N; i++) Pi_avg[i] /= training_count;
	for(int i = 0; i < N; i++) for(int j = 0; j < N; j++) A_avg[i][j] /= training_count;
	for(int i = 0; i < N; i++) for(int j = 0; j < codebook_size; j++) B_avg[i][j] /= training_count;
}


void dc_remover(char src[], char dest[])
{
	FILE *fp1 = fopen(src,"r");
	FILE *fp2 = fopen(dest,"w");

	if(!fp1){
		printf("dc: Can't read file!\n");
		return;
	}
		
	long double energy_sum = 0,cntr=0;
	long double sample_count = 0;
	long double temp, mean, new_value;
	while(fscanf(fp1,"%Lf\n",&temp)!=EOF){
		sample_count++;
		energy_sum += temp;
	}

	mean = energy_sum/sample_count;
	fseek(fp1,0L,SEEK_SET);

	while(fscanf(fp1,"%Lf\n",&temp)!=EOF){
		new_value = temp - mean;
		fprintf(fp2, "%Lf\n",new_value);
	}
	fclose(fp1);
	fclose(fp2);
}

int normalize_waveform(char in_filepath[])
{

	FILE *fp1 = fopen(in_filepath,"r");
	FILE *fp2 = fopen("normalized.txt","w+");  

	if(!fp1){
		printf("normalize: Cannot open file");
		return -1;
	}
		
	// finding out the maximum energy value
	long double temp, max_energy;
	fscanf(fp1,"%Lf\n",&temp);
	max_energy = temp;
	
	int sample_number = 1;
	int max_sample = 1;
	while(fscanf(fp1,"%Lf\n",&temp) != EOF){
		sample_number++;	
		if(abs(temp)>abs(max_energy)){
			max_energy = temp;   
			max_sample = sample_number;
		}                        
	}

	// normalizing energy values
	fseek(fp1, 0, SEEK_SET);
	long double normalized;

	while(fscanf(fp1,"%Lf\n",&temp) != EOF){
		normalized = (temp/max_energy)*normalization_value;
		fprintf(fp2,"%Lf\n",normalized);	
	}

	fclose(fp1);
	fclose(fp2);
	return max_sample;			// peak index returned
	
}

/*

int* set_markers(int peak, int waveform_size)
{
	FILE *reader;
	//reader reads the speech waveform; and writer writes the sample numbers where we possibly put the markers
	int i, start_marker = 0, end_marker = 0; 
	int sample_count = 0, samples_read = 0, frame_number = 0;
	static int computed_markers[2];
	long double samples[eval_samples * threshold_frame_set];
	long double curr_value, prev_frame_st_energy, curr_frame_st_energy, st_energy = 0;


	while(sample_count < waveform_size)
	{
		samples[samples_read] = waveform[sample_count];

		samples_read++;
		sample_count++;
		if(samples_read == (eval_samples * threshold_frame_set))
		{	

			frame_number += threshold_frame_set;
			//calculating short-term energy
			for(i=0;i<(eval_samples * threshold_frame_set);i++)
				st_energy += samples[i]*samples[i];

			st_energy = st_energy / (threshold_frame_set * eval_samples);

			if(frame_number == threshold_frame_set)	// for first 3 frames, so no comparison for threshold with any other frame
				prev_frame_st_energy = st_energy;

				
			else
			{
				curr_frame_st_energy = st_energy;
				if(curr_frame_st_energy > (start_frame_energy_threshold * prev_frame_st_energy))
				{
					start_marker = (frame_number - threshold_frame_set) * eval_samples;
					break;  
				}
				else
					prev_frame_st_energy = curr_frame_st_energy;
			}
			//resetting parameters for next frame
			samples_read=0;
			st_energy = 0;
			memset(samples,0,sizeof(samples));

		}
			
	}


	memset(samples,0,sizeof(samples));
	samples_read = 0;
	st_energy = 0;
	frame_number = 0;
	sample_count = waveform_size - 1;
	prev_frame_st_energy = 0;
	curr_frame_st_energy = 0;

	while(sample_count > peak)
	{

		samples[samples_read] = waveform[sample_count];
		samples_read++;
		sample_count--;
		if(samples_read == (threshold_frame_set * eval_samples))
		{	
			frame_number += threshold_frame_set;
			//calculating short-term energy
			for(i=0;i<eval_samples;i++)
				{
					st_energy += samples[i]*samples[i];
					if(st_energy <0)
						{ printf("%d",i);
							printf("%Lf ", st_energy);
					} 
				}
			st_energy = st_energy / (eval_samples * threshold_frame_set);
			//printf("\nst_energy till sample %d is %Lf", sample_count, st_energy);

			if(frame_number == threshold_frame_set)	// for first frame, so no comparison for threshold with any other frame
				prev_frame_st_energy = st_energy;

			else
			{
				curr_frame_st_energy = st_energy;
				if(curr_frame_st_energy > (end_frame_energy_threshold * prev_frame_st_energy))
				{
					end_marker = waveform_size - (frame_number * eval_samples);
					break;  
				}
				else
					prev_frame_st_energy = curr_frame_st_energy;
			}
			//resetting parameters for next frame
			samples_read=0;
			st_energy = 0;
			memset(samples,0,sizeof(samples));

		}
		
	}


				
	computed_markers[0] = start_marker;
	computed_markers[1] = end_marker;
	return computed_markers;
	

} */


void finding_the_speech(int start_marker, int end_marker){

	FILE *fp1 = fopen("normalized.txt","r");
	FILE *fp2 = fopen("word.txt","w");  
	long double temp;
	int counter = 0;

	while(fscanf(fp1,"%Lf\n",&temp)!= EOF)
	{
		counter++;
		if((counter>=start_marker)&&(counter<=end_marker))
			fprintf(fp2, "%Lf\n", temp);
		if(counter==end_marker)
			break;
	}
	fclose(fp1);
	fclose(fp2);
}

int samples_count(){
	long double temp;
	int counter = 0;
	FILE *fp = fopen("word.txt","r");  
	
	while(fscanf(fp,"%Lf\n",&temp)!= EOF)
		counter++;

	fclose(fp);
	return counter;

}

void extract_frame(int i, long double s[])
{
	FILE *reader;
	//reader reads the speech waveform; and writer writes the sample numbers where we possibly put the markers
	
	reader = fopen("word.txt","r");

	if(reader==NULL)
		printf("markers: Can't open file!");
	else
	{
		long double curr_value;
		int sample_count = 0, frame_start = 0, pos = 0;

		if(i == 0) 	// for the first frame no stride is required
		{
			while(sample_count < eval_samples)
			{
				//extracting our samples
				fscanf(reader,"%Lf\n",&curr_value);
				s[pos] = curr_value;
				pos++;
				sample_count++;
			}
		}

		else 		// for subsequent frames
		{
			// move pointer up to start of the frame
			
			while(frame_start < (stride * i))
			{
				fscanf(reader,"%Lf\n",&curr_value);
				frame_start++;	
			}	

			// now extract the signal elements
			while(sample_count < eval_samples)
			{
				//extracting our samples
				fscanf(reader,"%Lf\n",&curr_value);
				s[pos] = curr_value;
				pos++;
				sample_count++;
			}

		}

	}

	fclose(reader);
}

void calculate_ri(long double s[], long double r[]){

	FILE  *fp = fopen( "r_i.txt","w");

	long double r_i;
	for(int k=0; k<past_samples+1; k++)	{	//since we need values from r[0] to r[past_samples]
		r_i = 0;
		for(int m=0; m<=(eval_samples-1-k); m++)
			r_i += s[m]*s[m+k];

		r[k] = r_i;
		fprintf(fp, "%Lf\n", r[k]);
	}
	fclose(fp);
}


int calculate_ai(long double r[], long double a[]) 
{
	long double e[past_samples+1] = {0}, k[past_samples+1] = {0}, b[past_samples+1][past_samples+1];
	
	FILE  *fp = fopen("coefficients.txt","w");

	e[0] = r[0];
	if(e[0]==0){
		fclose(fp);
		return 0;
	}

	long double sum;
	for(int i=1; i<=past_samples; i++){
		sum = 0;
		for(int j=1; j<=(i-1); j++)
			sum += b[i-1][j]*r[i-j];

		k[i] = (r[i]-sum)/e[i-1];
		
		b[i][i] = k[i];

		for(int j=1; j<=i-1; j++)
			b[i][j] = b[i-1][j] - (k[i]*b[i-1][i-j]);

		e[i] = (1 - k[i]*k[i]) * e[i-1];
	}

	for(int i=1; i<=past_samples; i++)
	{
		a[i] = b[past_samples][i];
		fprintf(fp, "%Lf\n", a[i]);
	}

	fclose(fp);
	return 1;
}


void calculate_ci(long double a[], long double c[])
{
	FILE *reader, *writer;

	FILE *fp2 = fopen("cepstral_coefficients.txt","w");
	

	int i=1;			
	long double curr_value;

	// not needed if we do not use raised sine window

	FILE *fp1 = fopen("rsw_values.txt","r");
	long double rsw[past_samples+1] = {0};

	while(fscanf(fp1, "%Lf\n", &curr_value)!= EOF)
	{
		rsw[i] = curr_value;				//rsw values stored from index 1
		i++;
	}

	int k, q;
	long double sum, num, l_i, l_k;
	q = past_samples;
	for(i=1; i<=q; i++)
	{
		sum = 0;
		l_i = (long double)i;
		for(k=1; k<=(i-1); k++)
			{
				l_k = (long double)k;
				num = (l_k/l_i)*c[k]*a[i-k];
				//printf("\nnum=%Lf",num);
				sum += num;
			}

		//printf("\nsum=%Lf",sum);
		c[i] = a[i]+sum;		
	}


	// use this when you want to apply raised sine window on c[i] values
	
	for(i=1; i<=q; i++){
		c[i] = c[i]*rsw[i];			//use this only if you want to use RS window, hopefully will give correct results now!
		fprintf(fp2, "%Lf\n", c[i]);
	}

	fclose(fp1);					//use this only if you want to use RS window, hopefully will give correct results now!
	fclose(fp2);
}


void translate_to_observation_sequence(long double c[], int obs_seqs[], int framenum)
{
	int i, j, codebook_index;
	long double diff, total_distance, min_distance;
	long double total_distances[codebook_size];


	for(j = 0; j < codebook_size; j++)		// codebook indices are from 0 to codebook_index-1
	{
		total_distance = 0;
		for(i = 1; i <= past_samples; i++) // c array starts from index 1 to p, other arrays here start from 0
		{
			diff = c[i] - codebook[j][i-1];
			total_distance += tw[i-1] * diff * diff;
		}

		total_distances[j] = total_distance;
	}


	codebook_index = 0;
	min_distance = total_distances[codebook_index];
	
	for(i = 1; i < codebook_size; i++)
	{
		if(total_distances[i]<min_distance)
			{
				codebook_index = i;
				min_distance = total_distances[i];
			}
	}

	// since obs. seq. has range [1, codebook_size], and array is from [0, codebook_size-1], so adding 1. This
	// completes the translation of cepstral coeff. vector to obs eq. element for the current frame
	obs_seqs[framenum] = codebook_index + 1;

}

void show_obs_seq(int obs_seqs[], int framecount){
	int i;
	printf("\nObs. seq: \n");
	for(i = 0; i < framecount; i++)
		printf("%d ", obs_seqs[i]);
}


void process_digit(int training, char curr_digit[], int utterance_num, char filenum[], long double s[], long double r[], long double a[], long double c[], int avg_num, int digit){
	char display[msgsize], in_filepath[path_buffer_size];
	int success, framenum, waveform_size, i;
	int obs_seqs[max_frame_count];

	memset(obs_seqs, -1, sizeof(obs_seqs));

	//filename creation
	if(curr_digit[0] != '-'){
		_snprintf(in_filepath, sizeof(in_filepath), "%s%s%s%s%s%s", data_path, header, curr_digit, "_", filenum, ".txt");
		dc_remover(in_filepath,"dc_removed.txt");
	}else{
		dc_remover("input_file.txt","dc_removed.txt");
	}
	//printf("\n\nStarting training for digit %d observation sequence %d", digit, utterance_num);
	printf("\n---------------------------------------------------------------------------------------------------------------------------------------------");
	if(training)
		printf("\n%d : %d", digit, utterance_num);
	else if(curr_digit[0] != '-')
		printf("\n%d : %d", digit, utterance_num);

	
	int framecount, temp, peak_index;
	int* marker_ptr;
	long double max_energy, thresh;

	peak_index = normalize_waveform("dc_removed.txt");    // returns max_noramlized_value_index
	waveform_size = read_waveform();
	
	//for fixed obs seq length of 85
	int start_marker, end_marker;
	start_marker = peak_index - needed_samples/2;
	end_marker = peak_index + needed_samples/2 -1;
	finding_the_speech(start_marker, end_marker);



	//Step 3: Extracting the frame of 320 samples-----------------------------------------------------

	// calculating number of frames in the extracted word
	temp = ((samples_count() - eval_samples) / stride) + 1;
	framecount = (temp < max_frame_count) ? temp : max_frame_count; 
	
	// now that we know the no. of frames, the same number would be the length of the observation sequence
	for(framenum=0; framenum < framecount; framenum++){
		//resetting all buffers
		memset(s, 0, sizeof(s));
		memset(r, 0, sizeof(r));
		memset(a, 0, sizeof(a));
		memset(c, 0, sizeof(c));
		
		extract_frame(framenum, s);

		

		//getting the C_i values--------------------------------------
		calculate_ri(s, r);
		success = calculate_ai(r, a);
		calculate_ci(a, c);

		// now from the c_i's, translate to get the observation sequence
		translate_to_observation_sequence(c, obs_seqs, framenum);
		
	}
	
	printf("\n observation sequence: ");
	for(i = 0; i < framecount; i++){
		obs_seqs[i] = obs_seqs[i] - 1;	
		printf("%d ",obs_seqs[i]);
	}
	
	
		
	
	//now final part------------------------------------------------------------
	if(training){
		init_HMM_params();
		load_initial_model(framecount, avg_num, digit);
		HMM_model(framecount, obs_seqs);
		addIntoAverage();
	}else{
		long double prob_obs_seq, max_prob;
		int max_prob_index= 0;

		init_HMM_params();
		load_generated_model(framecount, 0);
		prob_obs_seq = solnProblemOne(framecount, obs_seqs);
		max_prob = prob_obs_seq;
		//printf("\n %d : %e  ", 0, prob_obs_seq);
		for(i = 1; i < digits_count; i++){
			init_HMM_params();
			load_generated_model(framecount, i);
			//show_model();
			prob_obs_seq = solnProblemOne(framecount, obs_seqs);
			printf("\n %d : %d  ", i, order_check(prob_obs_seq));

			if(prob_obs_seq > max_prob){
				max_prob = prob_obs_seq;
				max_prob_index = i;
			}
		}

		if(curr_digit[0] != '-'){
			printf("\nActual digit: %s\t Recognized digit: %s", all_digits[digit], all_digits[max_prob_index]);	
			if(digit == max_prob_index){
				digit_match++;
				overall_match++;
			}
		}
		else
			printf("\n\nRecognized digit : %s",all_digits[max_prob_index]);

	}
		
}



int _tmain(int argc, _TCHAR* argv[])
{
	char display[msgsize];
	char in_filepath[path_buffer_size];
	char num[3], ch_time[10];

	long double s[eval_samples];	// 320 samples evaluate as of now
	long double r[past_samples+1];	// p+1 values of R_i
	long double a[past_samples+1];	// p values of A_i stored from index 1
	long double c[past_samples+1];	// p values of C_i stored from index 1

	// read from files first and store tokura's weights and codebook in an array once and for all

	//storing tokura's weights
	FILE *fp_tw = fopen("tokura_wts.txt", "r");
	FILE *fp_cb = fopen("codebook.txt", "r");

	int i = 0;
	long double temp;
	while(fscanf(fp_tw, "%Lf\n", &temp) != EOF)
		tw[i++] = temp;	

	// storing codebook
	for(i = 0; i < codebook_size; i++)
		for(int j = 0; j < past_samples; j++){
			fscanf(fp_cb, "%Lf", &temp);
			codebook[i][j] = temp;
		}

	fclose(fp_tw);
	fclose(fp_cb);

	//-------------------------------------------------------------------------------------------------------------
	while(1){
		int option;
		printf("\nModule Menu: ");
		printf("\n0. To start the training. \n1. Recognize on own test file \n2. Recognize on live recording \n3. Exit\n\n");
		scanf("%d",&option);
		//declaration-----------------------------
		int training;
		long double overall_accuracy, digit_accuracy;
		switch(option){
		case 0:
				training = 1;		// the variable only has boolean use here 
				for(int digit_num = 0; digit_num < digits_count; digit_num++){
					for(int avg_num = 0; avg_num < averaging_count; avg_num++){	// averaging out the models for this digit
						init_HMM_params_global();	// everything related to HMM initialized for this digit
						printf("\nAverage number : %d", avg_num+1);
						for(int utterance_num=1; utterance_num <= training_count; utterance_num++){		// since filenames start with utterance no. 1 
							memset(num,0,3);
							sprintf(num,"%d",utterance_num);
							process_digit(training, all_digits[digit_num], utterance_num, num, s, r, a, c, avg_num, digit_num); //avg_num=0 => Inertia model. else taken from A_avg and B_avg.

						}
						getAverageModel();
						saveFinalModel(digit_num);
					}			
				}
				printf("\n\nFinished training !");
				break;

		case 1:
				//testing on pre-recorded input
				training = 0;
				overall_match=0;

				for(int digit_num=0; digit_num < digits_count; digit_num++){
					digit_match = 0;
					printf("\n\nStarting recognition system for digit %d\n\n", digit_num);

					for(int utterance_num=training_count+1; utterance_num <= training_count + testing_count; utterance_num++){
						memset(num,0,3);
						sprintf(num,"%d",utterance_num);
						process_digit(training, all_digits[digit_num], utterance_num, num, s, r, a, c, -1, digit_num);
					}
					digit_accuracy = ((long double)digit_match / (long double)testing_count) * 100;
					printf("\nAccuracy in recognizing %s : %Lf", all_digits[digit_num], digit_accuracy);
				}
				overall_accuracy = ((long double)overall_match / (long double)(testing_count * digits_count)) * 100;
				printf("\n\n Overall accuracy of the system : %Lf\n", overall_accuracy);
				break;		

		case 2:
				//testing on live recording
				training = 0;
				int timeslice;
				printf("\nEnter recording time (in integral seconds): ");
				scanf("%d",&timeslice);
					
				//resetting buffers
				memset(in_filepath,0,msgsize);
				memset(ch_time,0,sizeof(ch_time));
				memset(num,0,3);
				sprintf(ch_time,"%d",timeslice);

				_snprintf(in_filepath, sizeof(in_filepath), "%s %s %s %s","Recording_Module.exe", ch_time, "input_file.wav","input_file.txt");
				//Ex. - ("Recording_Module.exe 3 input_file.wav input_file.txt");

				system(in_filepath);
				process_digit(training, "-1", -1, "-1", s, r, a, c, -1, -1);  // -1 means unknown
				break;
		case 3:
				exit(0);
		default:
				printf("pls input the valid choice.\n");
		}
	}	
	

	getch();
	return 0;
}



