o1=3.5,
                 o2=6,
                 # o2=0.2,
                 # o1=0.1,
                 # o2=0.2,
                 o3=1.4
    allow overday charging false
                 reward scaling none (not divided by 100, is a good decision)
                 smoothing = 0.35 (needs to be bigger but then it can respond well to abrupt changes in env)
                 reducing smoothing - can reduce MSE, but will increase MPE because of the big spaces between ut and ut+1
                 increasing smoothing - possibly increase MPE, but also decrease MPE

                 if i increase smoothing window then i will take into account trend more general which can cause problems when there are radical changes based on recent trends
                 if i decrease smoothing window then i will have problems with rapidly changing trends too

                 this is so far the best model, regarding of MSE
                 still has problems with late charging - if

                 trained only on first 2 caltech garages

                 potentially increase o3 coeficient because even when smoothed part is small
                 and given energy is quite small too (15) it gives big u_t