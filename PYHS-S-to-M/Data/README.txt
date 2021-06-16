In this folder you will find data to use for machine learning the MCT glass transition of a hard sphere PY system.

It contains:
1. a folder called phi_data
2. a folder called Sk_data
3. a text file called k_array
4. three julia scripts
5. two images

Additional information:
1. 	In the phi_data folder, the main output data is stored. 
	It contains 10^5 textfiles each containing the intermediate scattering function at a volume fraction specified in the name of the file at kD=7.4.
	Each file contains two rows, the first row with the log10(t) data, and the second row containing the associated F(k,t). 
	The time points are randomly sampled between -10 and 10, differing for each file. The associated F is spline interpolated to correspond to the specific t-value.
	The volume fraction is sampled from a normal disstribution centred around the critical volume fraction with a standard deviation of 0.04
2. 	In the Sk_data folder, the main input data is stored. 
	It contains 10^5 textfiles each containing the static structure factor at a volume fraction specified in the name of the file.
	Each file with a structure factor corresponds to a single file with a intermediate scattering function, at the same volume fraction
3. 	The file k_array contains the k-values at which the structure factors are sampled
4. 	The julia scrips were called to generate this data, calculate the structure factor and solve the MCT equations
5.	The images are plots of the intermediate scattering functions at a small subset of the total dataset.  