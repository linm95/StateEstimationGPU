# StateEstimationGPU
This is a GPU program for the state estimation in power system

## introduction

For the monitoring and dispatching of power system, state estimation is a very important part. Its principle is to estimate the state of the power system with the data got from the SCADA, including the amplitude and phase angle of the voltage of nodes. The state estimation offers precise and reliable data for the subsequent computation of power flow. It can be regarded as a kind of data filtering and it sacrifices the redundancy for accuracy and can also be regarded as generalized power flow computation.

In nowadays state estimation system, SCADA updates data every millisecond which is much more fast than the speed of state estimation computation, therefore making the monitoring of power system slower and unable to response to the accident quickly, threating the stability of power system. So the speed-up of state estimation is of great significance for monitoring and dispatching for power system.
The process of state estimation involves many matrix related computation like mat-mat multiply, mat-vector multiply, forward and backward, which all are the computation with high parallelism. 

GPU is a kind of processer that specializes in parallel computation, so the introduction of GPU in state estimation can make great influence over the speed of computation.

##  The algorithm of state estimation based on fast decomposition

Considering the large size of actual power system, this algorithm overlooks some secondary factors to decrease the computing load and increase the computing speed.

It manly has two ways for simplification.

The decomposition of active power and reactive power. When a high voltage power system runs well, the couplings between voltage amplitude v and active power P, phase angle δ and reactive power Q are so weak that we can substitute the items ∆P/∆v and ∆Q/∆θ in the Jacob matrix H with 0. This will help us decrease the size of the matrix and requirement for RAM but increase the speed to get a convergent result.
  
Make the Jacob matrix constant. In the WLS algorithm, Jacob matrix changes every iteration but it changes slightly each time. If the Jacob matrix is regarded as constant, a convergent result can still be got without updating the Jacob matrix every iteration. It also saves the time for LU decomposition, thus increasing the speed.
  
The fast decomposition algorithm has the advantages of high speed, saving RAM, great convergence and the concrete algorithm is as followed

First, divide the state vector x into amplitude v and phase angle θ:

x=[■(θ@v)]

where the amplitude and phase angle of the reference node is v_0^ and θ_0^.

Then, divide the measurement vector z into active power and reactive power:

z=[■(z_α@z_r )]=[■(h_a (θ,v)@h_r (θ,v) )]=h(θ^ ,v^  )

The steps is as followed:

1. k=0, initialize the vector 〖v=v〗^0,θ=θ^0

2. According to formula:

A=v_0^4 [(-B_a )^T (-B_a )]

--where B_a equals to the reciprocal of branch reactance

B=v_0^2 [(-B_r )^T (-B_r )]

--where B_r equals to the imaginary part of branch admittance
		
3. LU decompose the A and B, getting L_A U_A,L_B U_B

-- A=L_A U_A

-- B=L_B U_B

4. According to:

α^k=[■((∂h_a^T)/∂θ&(∂h_r^T)/∂θ)][z-h(θ^(k-1),v^(k-1) )]

Compute the free vector α^k

5. According to:

A∆θ^k=α^k

Get the corrections ∆θ

6. According to:

b^k=[■((∂h_a^T)/∂v&(∂h_r^T)/∂v)][z-h(θ^(k-1),v^(k-1) )]

Compute the free vector b^k

7. According to:

B∆v^k=b^k

Get the corrections ∆v

8. Update the state vector θ^(k+1)=θ^k+∆θ, v^(k+1)=v^k+∆v

9. Check whether the accuracy requirement is satisfied. If so, output the result or get back to step 4.

## Implementation
This program is aimed at accelerating the state estimation in power system with GPU, implementing the complete computation of state estimation and realizing significant speed-up.

The features of this program lie in the facts that it combines the advantages of GPU and CPU. For the serial part of the program, it mainly utilizes the CPU whereas for the parallel part, it mainly uses the GPU. Meanwhile, the algorithm well be improved to fully excavate the parallelism.

First, the program is divided into initialization stage and iterating stage, where initialization stage corresponds to the stage 1-3, only executing once, and iterating stage corresponds to the stage 4-7, executing multiple times.

In both stages, program is divided into multiple modules based on their own computing features. Then each module will be matched with GPU or CPU according to their efficiency on the two platforms, which is called modulization.

The initialization stage includes modules of Jacob computation, information matrix computation, LU decomposition, AMD and matrix reordering.

The iterating stage includes modules of measurement function computation, LU backward and forward computation, free vector computation and checking & updating.

In the final program, the initialization module only executes once and the time consumption of communication between GPU and CPU is negligible. Therefore, some modules of this stage are on CPU while others on GPU. The iterating stage needs to be executed multiple times and all the modules are implemented on GPU.

## Case test
### introduction

The tests of the program is mainly about the large scale cases in power system. The cases are all from the power flow cases of the Matpower 6.0. The results of the power flow computation are mixed with some noise as the measurements of the state estimation. This approach can maximize the scale of the measurements and can reflect the efficiency of the program in worst cases.

The information of the test platform is as followed:
### Hardware:
GPU: NVIDA Tesla P100-PCIE-12GB
CPU: Intel Xeon CPU E3-1230 V5
### Software:
GPU: CUDA 8.0
CPU: Visual Studio 2012
The cases used for testing are mixed of multiple 9241 nodes case and the largest case is 140 thousand nodes. The details of the cases are in table 4.1.
### Table 4.1 Info of cases

|   Name   | Node | Branch | Measurements |
| -------- | ---- | ------ | ------------- |
| Case9241 | 9241  | 16049  | 41339       |
| Case18481 | 18481 | 32098  | 82677       |
| Case36961 | 36961 | 64196  | 165353      |
| Case73921 | 73921 | 128392 | 330705      |
| Case147841 | 147841| 256784 | 661409      |

## Correctness test
Compare the test result with the actual value, the info is in table 4.2.
 
### Table 4.2 Correctness of Cases
| Case | 9241 | 18481 | 36961 | 73921 | 147841 |
| -- | -- | -- | -- | -- | -- |
| Max amp diff | 0.00226 | 0.00227 | 0.00258 | 0.0023 | 0.002446 |
| Ave amp diff | 0.000352 | 0.000351 | 0.000461 | 0.00034 | 0.000387 |
| Amp diff SD | 0.00016 | 0.00014 | 0.00021 | 0.00019 | 0.00031 |
| Max ang diff | 0.0051 | 0.0056 | 0.00596 | 0.0063 | 0.0061 |
| Ave ang diff | 0.0011 | 0.0012 | 0.00118 | 0.0016 | 0.00111 |
| Ang diff SD | 0.0015 | 0.00173 | 0.0015 | 0.0021 | 0.0014 |
 
The correctness has no problem.

## Efficiency test

The efficiencies of the cases are in table 4.3 and graph 4.1.
### Table 4.3 Efficiency of cases
| Case|Initialization(ms)|Iterating(ms)|Sum(ms)|
|--|--|--|--|
| Case9241|43.877|40.521|84.398|
| Case18481|90.461|62.288|152.749|
| Case36961|169.201|98.886|268.087|
| Case73921|311.112|175.371|486.483|
| Case147841|631.049|323.506|954.555|
 
Graph 4.1 Case results

## Result analysis

For a 9241-node power system, the time for state estimation is only 84.4 ms, which is much faster than any actual state estimation program.
Add up the time on CPU and the time GPU of each cases respectively and the data is as followed.

### Table 4.3 Time for GPU/CPU

| Case|GPU time(ms)|CPU time(ms)|ratio(GPU/CPU)|
| -- | -- | -- | -- |
| 9241|52.731|31.692|1.664|
| 18481|83.579|69.168|1.237|
| 36961|130.480|137.609|0.948|
| 73921|223.528|262.984|0.851|
| 147841|404.327|550.228|0.735|
 
In the case 9241, the time consumption of GPU is much greater than that of CPU, but in case 147841, the time consumption of GPU is much less than that of CPU. With the growing of the size of the cases, the time consumption on GPU is getting less and less, which means the greater the computation size is, the more obvious the acceleration effect. This exactly reflects the advantage of GPU in the state estimation.

## Conclusion

From the test result, we can conclude that the speed of state estimation has got the level of sub-second. Considering that the size of actual power system is about 200 thousand nodes, the time for state estimation can be completed in 150 ms with this GPU program, which proves the acceleration effect of GPU on state estimation. 

## Reference

1. Chen, Y., M. Rice, K. Glaesemann, and Z. Huang. “Sub-Second State Estimation Implementation and Its Evaluation with Real Data [J].” In 2015 IEEE Power Energy Society General Meeting, 1–5, 2015.

2. Elizondo, M. A., Y. Chen, and Z. Huang. “Reliability Value of Fast State Estimation on Power Systems [J].” In PES T D 2012, 1–6, 2012.

3. Zhitong Yu, Ying Chen, Yankan Song, Shaowei Huang, "Comparison of parallel implementations of controls on GPU for transient simulation of power system [J]", Control Conference (CCC) 2016 35th Chinese, pp. 9996-10001, 2016, ISSN 1934-1768.

4. Yu Erkeng. State Estimation of Power System. Beijing: China WaterPower Press, 1985.
  
