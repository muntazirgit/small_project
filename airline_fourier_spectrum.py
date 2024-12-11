import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data
def fourier_series_approximation(x, y, n_terms):
    """
    Function to calculate the Fourier series approximation with the first 'n_terms' terms.
    x: the time period (e.g., months or days)
    y: the data values (e.g., passenger numbers)
    n_terms: number of terms in the Fourier series
    """
    T = len(x)  
    approx = np.zeros_like(y, dtype=np.float64)  

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    for n in range(1, n_terms + 1):
        a_n = (2 / T) * np.sum(y * np.cos(2 * np.pi * n * x / T))
        b_n = (2 / T) * np.sum(y * np.sin(2 * np.pi * n * x / T))

        approx += a_n * np.cos(2 * np.pi * n * x / T) + b_n * np.sin(2 * np.pi * n * x / T)

    return approx, a_n, b_n, T  

def calculate_power_spectrum(xf, yf):
    powers = np.abs(yf)**2  
    periods = 1 / xf  
    
    
    valid_idx = (xf > 0) & (periods >= 7) & (periods <= 365) 
    
    return periods[valid_idx], powers[valid_idx]

def plot_power_spectrum(periods, powers, student_id):
    fig, ax = plt.subplots(figsize=(10, 6))

   
    ax.plot(periods, powers, label='Power Spectrum', color='blue', marker='o')
    ax.set_title(f'Power Spectrum of Passenger Numbers\nStudent ID: {student_id}')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Power')
    
    
    ax.set_xscale('log')
    ax.set_xlim(7, 365)

    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    
    ax.legend()

    plt.tight_layout()
    plt.savefig('figure2.png')
    plt.show()

def plot_fourier(monthly_avg, student_id):
   
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    if isinstance(monthly_avg, np.ndarray):
        monthly_avg = pd.Series(monthly_avg, index=np.arange(1, 13))

    
    ax1.bar(monthly_avg.index, monthly_avg.values, color='skyblue', label='Monthly Avg Passengers')
    ax1.set_title(f'Monthly Average Passengers\nStudent ID: {student_id}')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Daily Passengers')
    ax1.set_xticks(np.arange(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    
    months = np.arange(1, 13)  
    n_terms = 8  
    fourier_approx, _, _, _ = fourier_series_approximation(months, monthly_avg, n_terms)

    ax1.plot(months, fourier_approx, label=f'Fourier Approximation (First {n_terms} Terms)', color='red', linestyle='--')

    
    ax1.legend()

    plt.tight_layout()
    plt.savefig('figure1.png')
    plt.show()

def plot_power_spectrum(periods, powers, student_id):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(periods, powers, label='Power Spectrum', color='blue', marker='o')
    ax.set_title(f'Power Spectrum of Passenger Numbers\nStudent ID: {student_id}')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Power')
    
    ax.set_xscale('log')
    ax.set_xlim(7, 365)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.legend()

    plt.tight_layout()
    plt.savefig('figure2.png')
    plt.show()


def calculate_total_passengers(data, year=2022):
    year_2022 = data[data.index.year == 2022]
    total_passengers = year_2022['Number'].sum()
    return total_passengers

def calculate_monthly_distribution(data):
    data['Month'] = data.index.month
    monthly_avg = data.groupby('Month')['Number'].mean()
    return monthly_avg


def calculate_main_period(xf, yf):
    main_period_idx = np.argmax(yf[1:]) + 1  
    threshold = 1e-10

    if abs(xf[main_period_idx]) > threshold:
        main_period = 1 / xf[main_period_idx]
    else:
        print("Warning: Frequency at main period index is too small!")
        main_period = float('inf')
        
    return main_period


def perform_fourier_transform(data):
    
    y = data['Number']
    N = len(y)
    T = 1.0  
    
   
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N//2]
    return xf, np.abs(yf[:N//2])


def calculate_monthly_distribution(data):
    data['Month'] = data.index.month
    monthly_avg = data.groupby('Month')['Number'].mean()
    return monthly_avg

def main():
    student_id = "22100478"  
    dataset_file = r'D:\tasks\airline7_-631351429.csv' 

    # Step A: Read data
    data = read_data(dataset_file)
    
    # Step B: Perform Fourier transform
    xf, yf = perform_fourier_transform(data)
    
    # Step C: Calculate monthly distribution
    monthly_avg = calculate_monthly_distribution(data)
    
    # Step D: Calculate total passengers for 2022
    X = calculate_total_passengers(data)
    
    # Step E: Calculate the main period in the power spectrum
    Y = calculate_main_period(xf, yf)
    
    # Step F: Plot the data
    plot_fourier(monthly_avg,student_id)

    periods, powers = calculate_power_spectrum(xf, yf)
    print("periods", periods, "powers", powers)
    plot_power_spectrum(periods, powers, student_id)


if __name__ == "__main__":
    main()
