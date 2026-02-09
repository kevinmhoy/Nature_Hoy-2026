# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
sys.path.append("/scratch/home/khoy/Programming/astroemperor/src/")
import numpy as np
import astroemperor as emp

#np.random.seed(1234)

'''
This script is set up to produce all the different sets of Keplerian fits showin in the work.
Set these to False to save time by skipping that specific run.
'''

K2_Open_Priors = True

K2_14_Day_Window = True
K2_70_Day_Window = True
K2_88_Day_Window = True
K2_115_Day_Window = True



#Make a list in the arbitrary order I used
included_runs = [K2_88_Day_Window, K2_115_Day_Window, K2_14_Day_Window, K2_70_Day_Window, K2_Open_Priors]

run_IDs = ['CD352722',
           ]

#like_max    -260.9897399176441

eng = 'reddemcee'
AM_ = False
keplerian_parameterisation = 1
dvals = []

#MCMC Setups, they're unique per run but there's no reason for them to be different at this stage
setups = [np.array([10, 128, 50000, 1]),
          np.array([10, 128, 50000, 1]),
          np.array([10, 128, 50000, 1]),
          np.array([10, 128, 50000, 1]),
          np.array([10, 128, 50000, 1]),
          ]

#We're set up to work on multiple data sets, but we only use one here
for run_ID in run_IDs:

    #Loop over the different runs
    for runno, setup in enumerate(setups):

        #If a run should be skipped, skip it
        if not included_runs[runno]:
            continue

        print(f'{runno=}')

        #Initialize object
        sim = emp.Simulation()

        #Backend setup
        sim.multiprocess_method = 0
        ntemps, nwalkers, nsweeps, nsteps = setup
        sim.use_c = False
        sim.keplerian_parameterisation = keplerian_parameterisation
        sim.save_backends = False  # TODO: test
        sim.cherry['cherry'] = True  # todo: test
        sim.cherry['median'] = True  # todo: ^
        sim.switch_constrain = False
        sim.constrain['method'] = 'None'

        sim.set_engine(eng)
        #sim.constrain_method = 'range'#, 'GM'
        #sim.constrain_sigma = 3

        #Set up extra MCMC parameters
        if eng=='reddemcee':
            sim.engine_config['setup'] = setup
            sim.engine_config['tsw_history'] = True
            sim.engine_config['smd_history'] = True

            sim.engine_config['adapt_tau'] = 100
            sim.engine_config['adapt_nu'] = 1.5
            sim.engine_config['adapt_mode'] = 0

            sim.run_config['burnin'] = 0.5  # its in niter (considers walkers!)  # 0.75
            

            #sim.engine_config['betas'] = list(np.geomspace(1, 0.001, setup[0]))
            #sim.engine_config['betas'] = [1.0, 0.8019, 0.6284, 0.4834,
            #                              0.3645, 0.2715, 0.2013, 0.1473,
            #                              0.1069, 0.07687, 0.05409, 0.03716,
            #                              0.0248, 0.0164, 0.0107, 0.006868,
            #                              0.004318, 0.002667, 0.001648, 0.001]

            #sim.engine_config['betas'] = [1.0, 0.7024, 0.5195, 0.3935,
            #                              0.2968, 0.2165, 0.1512, 0.1015,
            #                              0.06739, 0.0449, 0.03058, 0.02031,
            #                              0.01256, 0.006762, 0.002988, 0.001]

        # ------------------------

        #Not the mass of a star, this is the mass of the host BD
        sim.starmass = 0.0356 
        sim.starmass_err = 0.00208

        sim.instrument_names_RV = ['CRIRES']

        #The system has no acceleration and the orbital acceleration is negligible on our timescales
        sim.acceleration = 0
    
        if AM_:
            sim.switch_inclination = True

        # general conds
        sim.add_condition(['Offset 1', 'limits', [-150, 30]])
        sim.add_condition(['Jitter 1', 'limits', [0, 60]])
        sim.add_condition(['Jitter 1', 'prargs', [0, 10]])  # 30, 15

        #Limit the first Keplerian to the strong, longer-period signal
        sim.add_condition(['Period 1', 'limits', [150, 500]])
        sim.add_condition(['Amplitude 1', 'limits', [0.001, 500]])

        #Set up eccentricity priors, we try to keep these as free as possible but allowing free eccentricity allows for over-fitting.
        sim.add_condition([f'Ecc_sin 1', 'prior', 'Normal'])
        sim.add_condition([f'Ecc_cos 1', 'prior', 'Normal'])
        sim.add_condition([f'Ecc_sin 1', 'prargs', [0, 0.2/np.sqrt(2)]])
        sim.add_condition([f'Ecc_cos 1', 'prargs', [0, 0.2/np.sqrt(2)]])
        
        #Giving the initial positions near where we know the best-fit solution will be helps to maintain stable statistics
        sim.add_condition(['Period 1', 'init_pos', [169.5, 170.0]])
        sim.add_condition(['Amplitude 1', 'init_pos', [239.8, 240.0]])
        sim.add_condition(['Phase 1', 'init_pos', [2.01, 2.02]])

        sim.add_condition(['Offset 1', 'init_pos', [-62.1, -62.0]])
        sim.add_condition(['Jitter 1', 'init_pos', [0, 5]])
                

        #We keep the same eccentricity priors for the second signal, and set them up here as they're the same for all runs.
        sim.add_condition([f'Ecc_sin 2', 'prior', 'Normal'])
        sim.add_condition([f'Ecc_cos 2', 'prior', 'Normal'])
        sim.add_condition([f'Ecc_sin 2', 'prargs', [0, 0.2/np.sqrt(2)]])
        sim.add_condition([f'Ecc_cos 2', 'prargs', [0, 0.2/np.sqrt(2)]])

        #Run-specific settings blocks

        # ~88 day period block
        if runno==0:
            sim.add_condition(['Period 2', 'limits', [80, 100]])
            sim.add_condition(['Period 2', 'init_pos', [86, 88]])
            #sim.add_condition(['Period 3', 'limits', [1.5, 100]])
            sim.add_condition(['Amplitude 2', 'limits', [0.001, 500]])
            sim.add_condition(['Amplitude 2', 'init_pos', [55, 60]])

            #Known initial temperatures based on previous run, stabilizes statistics
            sim.engine_config['betas'] = [1.0, 0.5761, 0.3286, 0.1873, 0.108, 0.06172, 0.03489, 0.01882, 0.008387, 0.002111]

            #We get the 0-satellite and 1-satellite fits later in another block, so we only do the 2-satellite model here
            run_k_min = 2
            run_k_max = 2 
                              
        # ~115day period block
        if runno==1:
            sim.add_condition(['Period 2', 'limits', [100, 120]])
            sim.add_condition(['Period 2', 'init_pos', [109, 111]])
            sim.engine_config['betas'] = [1.0, 0.6344, 0.4004, 0.2422, 0.1426, 0.07853, 0.04108, 0.02024, 0.008396, 0.002111]
                        
            sim.add_condition(['Amplitude 2', 'limits', [0.001, 500]])
            sim.add_condition(['Amplitude 2', 'init_pos', [55, 60]])

            run_k_min = 2
            run_k_max = 2

        # ~14 day period block
        if runno==2:
            sim.add_condition(['Period 2', 'limits', [10, 20]])
            sim.add_condition(['Period 2', 'init_pos', [13, 14]])

            sim.add_condition(['Amplitude 2', 'limits', [0.001, 500]])
            sim.add_condition(['Amplitude 2', 'init_pos', [55, 60]])

            sim.engine_config['betas'] = [1.0, 0.6099, 0.3651, 0.224, 0.1338, 0.07218, 0.03796, 0.02057, 0.009301, 0.002111]

            run_k_min = 2
            run_k_max = 2

        # ~70 day period block
        if runno==3:
            sim.add_condition(['Period 2', 'limits', [65, 75]])
            sim.add_condition(['Period 2', 'init_pos', [69, 71]])
                        
            sim.add_condition(['Amplitude 2', 'limits', [0.001, 500]])
            sim.add_condition(['Amplitude 2', 'init_pos', [55, 60]])

            sim.engine_config['betas'] = [1.0, 0.6125, 0.3595, 0.2044, 0.1139, 0.06161, 0.03455, 0.01932, 0.008702, 0.002111]

            run_k_min = 2
            run_k_max = 2
                    
        # Open 2nd signal priors block
        if runno==4:
            sim.add_condition(['Period 2', 'limits', [1, 140]])
                        
            sim.add_condition(['Amplitude 2', 'limits', [0.001, 500]])
            sim.add_condition(['Amplitude 2', 'init_pos', [55, 60]])

            sim.engine_config['betas'] = [1.0, 0.5766, 0.3431, 0.2041, 0.1177, 0.06451, 0.0345, 0.01789, 0.007544, 0.002111]

            #Now we run the fits involving a lower number of satellites.
            run_k_min = 0
            run_k_max = 2



        model_ID = 'WN_2'
        if model_ID=='WN_1':
            logloc = 'datalogs/CD352722/CD352722_WN_1.txt'
            #run_k = 1
        if model_ID=='WN_2':
            logloc = 'datalogs/CD352722/CD352722_WN_2.txt'
            #run_k = 1





        # some options to speed up the test
        # PLOT OPTIONS
        if True:
            sim.plot_posteriors['plot'] = True
            sim.plot_posteriors['modes'] = [0]
            sim.plot_posteriors['temps'] = [0]
            sim.plot_posteriors['format'] = 'png'

            sim.plot_rates['window'] = 10
            sim.plot_histograms['plot'] = False
            
            sim.plot_trace['plot'] = True
            sim.plot_trace['format'] = 'svg'
            if eng == 'dynesty_dynamic':
                sim.plot_trace['modes'] = [0, 1, 3]  # dynesty
            else:
                sim.plot_trace['modes'] = [3]

            sim.plot_betas['plot'] = True
            sim.plot_rates['plot'] = True
            sim.plot_rates['window'] = 10


            sim.plot_keplerian_model['uncertain'] = False
            #sim.plot_keplerian_model['hist'] = False
            sim.plot_all['format'] = 'svg'
            sim.plot_all['paper_mode'] = True


        sim.load_data(run_ID)

        #sim.switch_AM = AM_
        sim.autorun(run_k_min, run_k_max)

        
        def get_enits(disc=sim.reddemcee_discard):
            tau = sim.sampler.backend[0].get_autocorr_time(tol=0,
                                                        discard=disc)[0]
            eff = 100/tau
            ncalls = int(ntemps*nwalkers*(nsweeps-disc)*nsteps)
            nits = np.array(ncalls)/np.array(sim.time_run)
            enits = nits/tau

            return nits, eff, enits


        print(f'time        {sim.time_run}')
        print(f'evidence    {sim.evidence}')
        print(f'like_max    {sim.like_max}')

        
        hs = []
        effs = []

        if eng == 'reddemcee':
            print(f'ti  {sim.sampler.get_evidence_ti(discard=sim.reddemcee_discard)[:2]}')
            print(f'ss  {sim.sampler.get_evidence_ss(discard=sim.reddemcee_discard)}')
            print(f'h   {sim.sampler.get_evidence_hybrid(discard=sim.reddemcee_discard)}')
            print(f'eff {get_enits()}')
            for i in range(len(dvals)):
                dval = dvals[i]
                if nsweeps > dval:
                    hs.append([sim.sampler.get_evidence_hybrid(discard=dval)])
                    effs.append([get_enits(dval)])
                    print(f'h{dval}     {hs[i]}')
                    print(f'eff{dval}   {effs[i]}')


        with open(logloc, 'a') as f:
            f.write(f'''
{eng}  N planets: {run_k_max}
{sim.saveplace_run}
{setup}
time        {sim.time_run}
evidence    {sim.evidence}
like_max    {sim.like_max}
ti          {sim.sampler.get_evidence_ti(discard=sim.reddemcee_discard)[:2]}
ss          {sim.sampler.get_evidence_ss(discard=sim.reddemcee_discard)}
h           {sim.sampler.get_evidence_hybrid(discard=sim.reddemcee_discard)}
eff         {get_enits()}
''')
            for i in range(len(dvals)):
                dval = dvals[i]
                f.write(f'''
eff{dval}   {effs[i]}
h{dval}     {hs[i]}
''')



