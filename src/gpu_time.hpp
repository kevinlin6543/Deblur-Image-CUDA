/* 
	Header file for GPU timing 
 	Based on https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/ 

	Usage:
		gpu_time gt;
		gt.begin();
		[stuff]
		gt.end();
		cout << "Elapsed time: " << gt.elap_time() << endl;
		
 */

class gpu_time {
	cudaEvent_t start, stop;

	/* Constructor */
	gpu_time()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	/* Destructor */
	~gpu_time()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	/* Begin the timer */
	void begin()
	{
		cudaEventRecord(start, 0);
	}

	/* Stop the timer */
	void end()
	{
		cudaEventRecord(stop, 0);
	}

	/* Calculate the total elapsed time */
	float elap_time()
	{
		float t = 0;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&t, start, stop);
		return t;
	}
};
