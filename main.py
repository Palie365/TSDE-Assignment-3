import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)

data1_df = pd.read_csv('data_tsde_assignment_3_part_1.csv')
cols = data1_df.columns
T1 = data1_df.shape[0]

def monteCarlo_part1():
    M = 1000
    T = [100, 500, 1000] ####### Code is written so that path lengths T can easily be added/removed/changed
    
    sim_betas = {}
    sim_tstats = {}
    sim_Rsquareds = {}
    
    print(f"--- PART 1.1: Spurious regression simulation study with M = {M} ---")

    for T in T:
        beta_hats = []
        t_stats = []
        R_squareds = []
        for m in range(M):
            nu = stats.norm.rvs(loc=0.0, scale=1.0, size=T)
            omega = stats.norm.rvs(loc=0.0, scale=1.0, size=T)
            
            X = np.zeros(T)
            Y = np.zeros(T)
            for t in range(1,T):
                X[t] = X[t-1] + nu[t]
                Y[t] = Y[t-1] + omega[t]
            
            X = sm.add_constant(X)
            model = sm.OLS(Y,X).fit()
            
            beta_hat, beta_se = model.params[1], model.bse[1]
            t_stat = beta_hat/beta_se
            R_squared = model.rsquared
            
            beta_hats.append(beta_hat)
            t_stats.append(t_stat)
            R_squareds.append(R_squared)
            
        sim_betas[T] = beta_hats
        sim_tstats[T] = t_stats
        sim_Rsquareds[T] = R_squareds

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(16, 6))

    sns.kdeplot(data=sim_betas, legend=False, ax=ax1).set(title=r'Distribution of estimated $\beta$s')
    sns.kdeplot(data=sim_tstats, legend=False, ax=ax2).set(title=r'Distribution of estimated $t$-statistics')
    sns.kdeplot(data=sim_Rsquareds, legend=True, ax=ax3).set(title=r'Distribution of estimated $R^2$s')
    fig.suptitle(f'Simulation study for M = {M}')
    #plt.savefig("MC_study_part1.png")
    plt.show()
                
def monteCarlo_part2():
    M = 1000
    T = [100, 500, 1000] ####### Code is written so that path lengths T can easily be added/removed/changed
    gamma = 1
    phi = 0.8
    
    print(f"--- PART 2.1: Simulation study (M = {M}) of cointegrated time series where" r" $\{X_t\}$ follows a random walk and a stable AR(1) process ---")

    '''
    X follows a random walk
    '''
    
    sim_betas = {}
    sim_tstats = {}
    sim_Rsquareds = {}
    
    for T in T:
        beta_hats = []
        t_stats = []
        R_squareds = []
        
        for m in range(M):
            nu = stats.norm.rvs(loc=0.0, scale=1.0, size=T)
            omega = stats.norm.rvs(loc=0.0, scale=1.0, size=T)
            
            X = np.zeros(T)
            Y = np.zeros(T)
            
            X[0] = 0
            Y[0] = omega[0]
            for t in range(1,T):
                X[t] = X[t-1] + nu[t]
                Y[t] = gamma*X[t-1] + omega[t]
            
            X = sm.add_constant(X)
            model = sm.OLS(Y,X).fit()
            
            beta_hat, beta_se = model.params[1], model.bse[1]
            t_stat = beta_hat/beta_se
            R_squared = model.rsquared
            
            beta_hats.append(beta_hat)
            t_stats.append(t_stat)
            R_squareds.append(R_squared)
            
        sim_betas[T] = beta_hats
        sim_tstats[T] = t_stats
        sim_Rsquareds[T] = R_squareds

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(16, 6))

    sns.kdeplot(data=sim_betas, legend=False, ax=ax1).set(title=r'Distribution of estimated $\beta$s')
    sns.kdeplot(data=sim_tstats, legend=False, ax=ax2).set(title=r'Distribution of estimated $t$-statistics')
    sns.kdeplot(data=sim_Rsquareds, legend=True, ax=ax3).set(title=r'Distribution of estimated $R^2$s')
    #plt.savefig("MC_study_rw_part2.png")
    fig.suptitle(f'Simulation study for M = {M}, ' r'$\{X_t\}$ follows a random walk')
    plt.show()   
    
    '''
    X follows a stable AR(1)
    '''
    
    T = [100, 500, 1000] ####### Code is written so that path lengths T can easily be added or removed

    sim_betas = {}
    sim_tstats = {}
    sim_Rsquareds = {}
    
    for T in T:
        beta_hats = []
        t_stats = []
        R_squareds = []
        
        for m in range(M):
            nu = stats.norm.rvs(loc=0.0, scale=1.0, size=T)
            omega = stats.norm.rvs(loc=0.0, scale=1.0, size=T)
            
            X = np.zeros(T)
            Y = np.zeros(T)
            
            X[0] = 0
            Y[0] = omega[0]
            for t in range(1,T):
                X[t] = phi*X[t-1] + nu[t]
                Y[t] = gamma*X[t-1] + omega[t]
            
            X = sm.add_constant(X)
            model = sm.OLS(Y,X).fit()
            
            beta_hat, beta_se = model.params[1], model.bse[1]
            t_stat = beta_hat/beta_se
            R_squared = model.rsquared
            
            beta_hats.append(beta_hat)
            t_stats.append(t_stat)
            R_squareds.append(R_squared)
            
        sim_betas[T] = beta_hats
        sim_tstats[T] = t_stats
        sim_Rsquareds[T] = R_squareds

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(16, 6))

    sns.kdeplot(data=sim_betas, legend=False, ax=ax1).set(title=r'Distribution of estimated $\beta$s')
    sns.kdeplot(data=sim_tstats, legend=False, ax=ax2).set(title=r'Distribution of estimated $t$-statistics')
    sns.kdeplot(data=sim_Rsquareds, legend=True, ax=ax3).set(title=r'Distribution of estimated $R^2$s')
    #plt.savefig("MC_study_stat_part2.png")
    fig.suptitle(f'Simulation study for M = {M}, ' r'$\{X_t\}$ follows a stable AR(1) ($\phi$'f' = {phi})')
    plt.show()   

def part1_2(stock1, stock2, lags):
    print(f"--- PART 1.2: Data Plots of {stock1}, {stock2} and corresponding ACFs ---")
    data_stock1 = data1_df[stock1]
    data_stock2 = data1_df[stock2]

    # ts plot stock1
    plt.figure(figsize=(10, 4))
    plt.plot(data_stock1, color='#4682b4', linewidth=1.5)
    plt.title(f"Daily {stock1} Stock Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.1)
    #plt.savefig("stock1_ts.png")
    plt.show()

    # ACF plot stock1
    plot_acf(data_stock1, lags=lags, alpha=0.05, zero=True, color='#cd5c5c')
    plt.title(f"SACF of {stock1} (lags 0-{lags})")
    plt.grid(True, alpha=0.1)
    #plt.savefig("stock1_sacf.png")
    plt.show()
    
    # ts plot stock2
    plt.figure(figsize=(10, 4))
    plt.plot(data_stock2, color='#4682b4', linewidth=1.5)
    plt.title(f"Daily {stock2} Stock Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.1)
    #plt.savefig("stock2_ts.png")
    plt.show()

    # ACF plot stock2
    plot_acf(data_stock2, lags=lags, alpha=0.05, zero=True, color='#cd5c5c')
    plt.title(f"SACF of {stock2} (lags 0-{lags})")
    plt.grid(True, alpha=0.1)
    #plt.savefig("stock2_sacf.png")
    plt.show()
    
def part1_3(p_max):
    print(f"--- PART 1.3: ADF test with/without constant/trend for each stock ---\n")
    for i in range(1,len(cols)):
        stock = cols[i]
        
        adf_n, p_val_n, used_lag_n, n_obs_n, crit_vals_n, icbest_n = adfuller(x=data1_df[cols[i]], maxlag=p_max, regression='n', autolag='BIC')   #  no constant/trend 
        adf_c, p_val_c, used_lag_c, n_obs_c, crit_vals_c, icbest_c = adfuller(x=data1_df[cols[i]], maxlag=p_max, regression='c', autolag='BIC')   # constant 
        adf_ct, p_val_ct, used_lag_ct, n_obs_ct, crit_vals_ct, icbest_ct = adfuller(x=data1_df[cols[i]], maxlag=p_max, regression='ct', autolag='BIC')   # constant and trend 

        BIC = [icbest_n, icbest_c, icbest_ct]
        ADF = [adf_n, adf_c, adf_ct]
        crit_vals = {}
        crit_vals[0], crit_vals[1], crit_vals[2] = crit_vals_n, crit_vals_c, crit_vals_ct
        model = ['no constant/trend', 'a constant', 'a constant and trend']
    
        index = np.argmin(BIC)
        
        print(f"Stock: {stock}")
        print(f"The best ADF specification is the model with {model[index]}. We have ADF = {ADF[index]} and the 10% critical value is {crit_vals[index]['10%']}.")
        if ADF[index] < crit_vals[index]['10%']:
            print(r'We reject the null hypothesis (unit root)' f" at a 10% significance level.\n")
        else:
            print(r'We fail to reject the null hypothesis (unit root)' f" at a 10% significance level.\n")        
        
def part1_4(p_max):
    # isolate data and first difference
    MS_data, EXXON_data = data1_df['MICROSOFT'], data1_df['EXXON_MOBIL']
    MS_diffdata, EXXON_diffdata = MS_data.diff().dropna(), EXXON_data.diff().dropna()
    
    
    # plot data and first difference
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4), constrained_layout=True)
    
    axes[0].plot(MS_data, color='#4682b4', linewidth=1.5)
    axes[0].set_title(f"Daily Microsoft Stock Price")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.1)
    
    axes[1].plot(MS_diffdata, color='#4682b4', linewidth=1.5)
    axes[1].set_title(f"First Difference of Microsoft Stock Price")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("First Difference of Price")
    axes[1].grid(True, alpha=0.1)
    plt.savefig("microsoft_diff.png")
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4), constrained_layout=True)
    
    axes[0].plot(EXXON_data, color='#4682b4', linewidth=1.5)
    axes[0].set_title(f"Daily Exxon Stock Price")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.1)
    
    axes[1].plot(EXXON_diffdata, color='#4682b4', linewidth=1.5)
    axes[1].set_title(f"First Difference of Exxon Stock Price")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("First Difference of Price")
    axes[1].grid(True, alpha=0.1)
    plt.savefig("exxon_diff.png")
    plt.show()
    
    
    # ADF for Microsoft diffdata
    adf_n, p_val_n, used_lag_n, n_obs_n, crit_vals_n, icbest_n = adfuller(x=MS_diffdata, maxlag=p_max, regression='n', autolag='AIC')   #  no constant/trend 
    adf_c, p_val_c, used_lag_c, n_obs_c, crit_vals_c, icbest_c = adfuller(x=MS_diffdata, maxlag=p_max, regression='c', autolag='AIC')   # constant 
    adf_ct, p_val_ct, used_lag_ct, n_obs_ct, crit_vals_ct, icbest_ct = adfuller(x=MS_diffdata, maxlag=p_max, regression='ct', autolag='AIC')   # constant and trend 

    AIC = [icbest_n, icbest_c, icbest_ct]
    ADF = [adf_n, adf_c, adf_ct]
    crit_vals = {}
    crit_vals[0], crit_vals[1], crit_vals[2] = crit_vals_n, crit_vals_c, crit_vals_ct
    model = ['no constant/trend', 'a constant', 'a constant and trend']
    
    index = np.argmin(AIC)
        
    print(f"Data: First Difference of Microsoft Stock Price")
    print(f"The best ADF specification is the model with {model[index]}. We have ADF = {ADF[index]} and the 1% critical value is {crit_vals[index]['1%']}.")
    if ADF[index] < crit_vals[index]['1%']:
        print(r'We reject the null hypothesis (unit root)' f" at a 1% significance level.\n")
    else:
        print(r'We fail to reject the null hypothesis (unit root)' f" at a 1% significance level.\n")
    
    # ADF for Exxon diffdata
    adf_n, p_val_n, used_lag_n, n_obs_n, crit_vals_n, icbest_n = adfuller(x=EXXON_diffdata, maxlag=p_max, regression='n', autolag='AIC')   #  no constant/trend 
    adf_c, p_val_c, used_lag_c, n_obs_c, crit_vals_c, icbest_c = adfuller(x=EXXON_diffdata, maxlag=p_max, regression='c', autolag='AIC')   # constant 
    adf_ct, p_val_ct, used_lag_ct, n_obs_ct, crit_vals_ct, icbest_ct = adfuller(x=EXXON_diffdata, maxlag=p_max, regression='ct', autolag='AIC')   # constant and trend 

    AIC = [icbest_n, icbest_c, icbest_ct]
    ADF = [adf_n, adf_c, adf_ct]
    crit_vals = {}
    crit_vals[0], crit_vals[1], crit_vals[2] = crit_vals_n, crit_vals_c, crit_vals_ct
    model = ['no constant/trend', 'a constant', 'a constant and trend']
    
    index = np.argmin(AIC)

    print(f"Data: First Difference of Exxon Mobil Stock Price")
    print(f"The best ADF specification is the model with {model[index]}. We have ADF = {ADF[index]} and the 1% critical value is {crit_vals[index]['1%']}.")
    if ADF[index] < crit_vals[index]['1%']:
        print(r'We reject the null hypothesis (unit root)' f" at a 1% significance level.\n")
    else:
        print(r'We fail to reject the null hypothesis (unit root)' f" at a 1% significance level.\n")
    
    # Investigate contemporaneous relation between Xt and Yt

    
    

if __name__ == "__main__": 
    monteCarlo_part1() 
    
    part1_2('INTEL', 'APPLE', lags=12)
    
    p_max = int(np.ceil(12*(T1/100)**0.25)) # rule of thumb for max lags
    part1_3(p_max) 
    
    part1_4(p_max)

    monteCarlo_part2() 

    
    
