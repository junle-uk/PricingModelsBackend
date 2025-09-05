import math
import numpy as np
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import norm
from typing import List, TypedDict

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://frontend:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class BlackScholesParams(BaseModel):
    spotPrice: float
    strikePrice: float
    timeToMaturity: float
    volatility: float
    riskFreeRate: float

class BlackScholesResult(BaseModel):
    callPrice: float
    putPrice: float
    d1: float
    d2: float

class GreeksCurveRequest(BaseModel):
    spotPrice: float
    strikePrice: float
    timeToMaturity: float
    volatility: float
    riskFreeRate: float
    rangeMin: float
    rangeMax: float
    steps: int
    curveType: str  # "gamma", "vega", "theta"

class GreeksCurveResponse(BaseModel):
    xValues: List[float]
    yValues: List[float]

class BinomialParams(BaseModel):
    spotPrice: float
    strikePrice: float
    timeToMaturity: float
    volatility: float
    riskFreeRate: float
    steps: int

class BinomialResult(BaseModel):
    callPrice: float
    putPrice: float

class MonteCarloParams(BaseModel):
    spotPrice: float
    strikePrice: float
    timeToMaturity: float
    volatility: float
    riskFreeRate: float
    simulations: int = 10000

class MonteCarloResult(BaseModel):
    callPrice: float
    putPrice: float

class Surface3DRequest(BaseModel):
    spotPrice: float
    strikePrice: float
    timeToMaturity: float
    volatility: float
    riskFreeRate: float
    spotMin: float
    spotMax: float
    timeMin: float
    timeMax: float
    spotSteps: int = 50
    timeSteps: int = 50
    model: str  # "black-scholes", "binomial", "monte-carlo"
    steps: int = 50  # For binomial
    simulations: int = 5000  # For monte-carlo

class Surface3DResponse(BaseModel):
    spotPrices: List[float]
    times: List[float]
    callPrices: List[List[float]]
    putPrices: List[List[float]]

class GreeksDict(TypedDict):
    delta: dict[str, float]
    gamma: float
    theta: dict[str, float]
    vega: float
    rho: dict[str, float]

def calculate_black_scholes(params: BlackScholesParams) -> BlackScholesResult:
    """Calculate Black-Scholes option prices"""
    if (params.timeToMaturity <= 0 or params.volatility <= 0 or 
        params.spotPrice <= 0 or params.strikePrice <= 0):
        return BlackScholesResult(callPrice=0.0, putPrice=0.0, d1=0.0, d2=0.0)
    
    # Calculate d1 and d2
    d1 = (math.log(params.spotPrice / params.strikePrice) + 
          (params.riskFreeRate + (params.volatility * params.volatility) / 2) * params.timeToMaturity) / \
         (params.volatility * math.sqrt(params.timeToMaturity))
    
    d2 = d1 - params.volatility * math.sqrt(params.timeToMaturity)
    
    # Calculate discount factor
    discount_factor = math.exp(-params.riskFreeRate * params.timeToMaturity)
    
    # Calculate call and put prices
    call_price = params.spotPrice * norm.cdf(d1) - params.strikePrice * discount_factor * norm.cdf(d2)
    put_price = params.strikePrice * discount_factor * norm.cdf(-d2) - params.spotPrice * norm.cdf(-d1)
    
    return BlackScholesResult(
        callPrice=max(0.0, call_price),
        putPrice=max(0.0, put_price),
        d1=d1,
        d2=d2
    )

def calculate_greeks(params: BlackScholesParams) -> GreeksDict:
    """Calculate all Greeks for Black-Scholes"""
    if (params.timeToMaturity <= 0 or params.volatility <= 0 or 
        params.spotPrice <= 0 or params.strikePrice <= 0):
        return {
            "delta": {"call": 0.0, "put": 0.0},
            "gamma": 0.0,
            "theta": {"call": 0.0, "put": 0.0},
            "vega": 0.0,
            "rho": {"call": 0.0, "put": 0.0}
        }
    
    d1 = (math.log(params.spotPrice / params.strikePrice) + 
          (params.riskFreeRate + (params.volatility * params.volatility) / 2) * params.timeToMaturity) / \
         (params.volatility * math.sqrt(params.timeToMaturity))
    d2 = d1 - params.volatility * math.sqrt(params.timeToMaturity)
    sqrtT = math.sqrt(params.timeToMaturity)
    discount_factor = math.exp(-params.riskFreeRate * params.timeToMaturity)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    
    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (params.spotPrice * params.volatility * sqrtT)
    
    # Theta (per year)
    term1 = -(params.spotPrice * norm.pdf(d1) * params.volatility) / (2 * sqrtT)
    call_theta = term1 - params.riskFreeRate * params.strikePrice * discount_factor * norm.cdf(d2)
    put_theta = term1 + params.riskFreeRate * params.strikePrice * discount_factor * norm.cdf(-d2)
    
    # Vega (for 1% change in volatility)
    vega = params.spotPrice * sqrtT * norm.pdf(d1) / 100.0
    
    # Rho (for 1% change in interest rate)
    call_rho = params.strikePrice * params.timeToMaturity * discount_factor * norm.cdf(d2) / 100.0
    put_rho = -params.strikePrice * params.timeToMaturity * discount_factor * norm.cdf(-d2) / 100.0
    
    return {
        "delta": {"call": call_delta, "put": put_delta},
        "gamma": gamma,
        "theta": {"call": call_theta / 365.0, "put": put_theta / 365.0},  # Daily theta
        "vega": vega,
        "rho": {"call": call_rho, "put": put_rho}
    }

def calculate_binomial(params: BinomialParams) -> BinomialResult:
    """Calculate Binomial option prices using Cox-Ross-Rubinstein"""
    if (params.timeToMaturity <= 0 or params.volatility <= 0 or 
        params.spotPrice <= 0 or params.strikePrice <= 0 or params.steps <= 0):
        return BinomialResult(callPrice=0.0, putPrice=0.0)
    
    dt = params.timeToMaturity / params.steps
    u = math.exp(params.volatility * math.sqrt(dt))
    d = 1.0 / u
    discount_factor = math.exp(-params.riskFreeRate * dt)
    p = (math.exp(params.riskFreeRate * dt) - d) / (u - d)
    
    # Build stock price tree
    stock_prices = [[0.0] * (params.steps + 1) for _ in range(params.steps + 1)]
    for i in range(params.steps + 1):
        for j in range(i + 1):
            stock_prices[i][j] = params.spotPrice * (u ** (i - j)) * (d ** j)
    
    # Calculate option values at expiration
    call_values = [[0.0] * (params.steps + 1) for _ in range(params.steps + 1)]
    put_values = [[0.0] * (params.steps + 1) for _ in range(params.steps + 1)]
    
    for j in range(params.steps + 1):
        stock_price = stock_prices[params.steps][j]
        call_values[params.steps][j] = max(0.0, stock_price - params.strikePrice)
        put_values[params.steps][j] = max(0.0, params.strikePrice - stock_price)
    
    # Backward induction
    for i in range(params.steps - 1, -1, -1):
        for j in range(i + 1):
            call_values[i][j] = discount_factor * (p * call_values[i + 1][j] + (1 - p) * call_values[i + 1][j + 1])
            put_values[i][j] = discount_factor * (p * put_values[i + 1][j] + (1 - p) * put_values[i + 1][j + 1])
    
    return BinomialResult(
        callPrice=call_values[0][0],
        putPrice=put_values[0][0]
    )

def calculate_monte_carlo(params: MonteCarloParams) -> MonteCarloResult:
    """Calculate option prices using Monte Carlo simulation (Geometric Brownian Motion)"""
    if (params.timeToMaturity <= 0 or params.volatility <= 0 or 
        params.spotPrice <= 0 or params.strikePrice <= 0):
        return MonteCarloResult(callPrice=0.0, putPrice=0.0)
    
    np.random.seed(42)  # For reproducibility
    dt = params.timeToMaturity
    discount_factor = math.exp(-params.riskFreeRate * dt)
    
    # Generate random paths using GBM
    z = np.random.standard_normal(params.simulations)
    stock_prices = params.spotPrice * np.exp(
        (params.riskFreeRate - 0.5 * params.volatility ** 2) * dt + 
        params.volatility * math.sqrt(dt) * z
    )
    
    # Calculate payoffs
    call_payoffs = np.maximum(stock_prices - params.strikePrice, 0.0)
    put_payoffs = np.maximum(params.strikePrice - stock_prices, 0.0)
    
    # Discount and average
    call_price = discount_factor * np.mean(call_payoffs)
    put_price = discount_factor * np.mean(put_payoffs)
    
    return MonteCarloResult(
        callPrice=float(call_price),
        putPrice=float(put_price)
    )

@app.get("/api/test")
async def test_endpoint_get():
    """Simple test endpoint (GET)"""
    return {"status": "ok", "message": "Backend is working!", "method": "GET", "timestamp": datetime.now().isoformat()}

@app.post("/api/test")
async def test_endpoint_post():
    """Simple test endpoint (POST)"""
    return {"status": "ok", "message": "Backend is working!", "method": "POST", "timestamp": datetime.now().isoformat()}

@app.post("/api/black-scholes", response_model=BlackScholesResult)
async def calculate_black_scholes_endpoint(params: BlackScholesParams) -> BlackScholesResult:
    """Calculate Black-Scholes option pricing"""
    return calculate_black_scholes(params)

@app.post("/api/greeks-curve", response_model=GreeksCurveResponse)
async def greeks_curve_endpoint(request: GreeksCurveRequest) -> GreeksCurveResponse:
    """Generate Greeks curve data (Gamma, Vega, or Theta)"""
    x_values: List[float] = []
    y_values: List[float] = []

    print("this api is hit")
    
    step = (request.rangeMax - request.rangeMin) / request.steps
    
    for i in range(request.steps + 1):
        if request.curveType == "gamma" or request.curveType == "vega":
            # Vary spot price
            spot_price = request.rangeMin + i * step
            params = BlackScholesParams(
                spotPrice=spot_price,
                strikePrice=request.strikePrice,
                timeToMaturity=request.timeToMaturity,
                volatility=request.volatility,
                riskFreeRate=request.riskFreeRate
            )
            x_values.append(spot_price)
        elif request.curveType == "theta":
            # Vary time to maturity
            time = request.rangeMin + i * step
            params = BlackScholesParams(
                spotPrice=request.spotPrice,
                strikePrice=request.strikePrice,
                timeToMaturity=time,
                volatility=request.volatility,
                riskFreeRate=request.riskFreeRate
            )
            x_values.append(time)
        else:
            continue
        
        greeks = calculate_greeks(params)
        if request.curveType == "gamma":
            y_values.append(greeks["gamma"])
        elif request.curveType == "vega":
            y_values.append(greeks["vega"])
        elif request.curveType == "theta":
            y_values.append(greeks["theta"]["call"])
    
    return GreeksCurveResponse(xValues=x_values, yValues=y_values)

def calculate_binomial_greeks(params: BinomialParams, price_change: float = 0.01, 
                              vol_change: float = 0.01, time_change: float = 1/365) -> GreeksDict:
    """Calculate Greeks for Binomial model using finite differences"""
    base_result = calculate_binomial(params)
    
    # Delta
    up_params = BinomialParams(
        spotPrice=params.spotPrice * (1 + price_change),
        strikePrice=params.strikePrice,
        timeToMaturity=params.timeToMaturity,
        volatility=params.volatility,
        riskFreeRate=params.riskFreeRate,
        steps=params.steps
    )
    down_params = BinomialParams(
        spotPrice=params.spotPrice * (1 - price_change),
        strikePrice=params.strikePrice,
        timeToMaturity=params.timeToMaturity,
        volatility=params.volatility,
        riskFreeRate=params.riskFreeRate,
        steps=params.steps
    )
    up_result = calculate_binomial(up_params)
    down_result = calculate_binomial(down_params)
    
    call_delta = (up_result.callPrice - down_result.callPrice) / (2 * params.spotPrice * price_change)
    put_delta = (up_result.putPrice - down_result.putPrice) / (2 * params.spotPrice * price_change)
    
    # Gamma
    mid_up_params = BinomialParams(
        spotPrice=params.spotPrice * (1 + price_change / 2),
        strikePrice=params.strikePrice,
        timeToMaturity=params.timeToMaturity,
        volatility=params.volatility,
        riskFreeRate=params.riskFreeRate,
        steps=params.steps
    )
    mid_down_params = BinomialParams(
        spotPrice=params.spotPrice * (1 - price_change / 2),
        strikePrice=params.strikePrice,
        timeToMaturity=params.timeToMaturity,
        volatility=params.volatility,
        riskFreeRate=params.riskFreeRate,
        steps=params.steps
    )
    mid_up_result = calculate_binomial(mid_up_params)
    mid_down_result = calculate_binomial(mid_down_params)
    gamma = (mid_up_result.callPrice - 2 * base_result.callPrice + mid_down_result.callPrice) / \
            ((params.spotPrice * price_change / 2) ** 2)
    
    # Theta
    time_up_params = BinomialParams(
        spotPrice=params.spotPrice,
        strikePrice=params.strikePrice,
        timeToMaturity=params.timeToMaturity - time_change,
        volatility=params.volatility,
        riskFreeRate=params.riskFreeRate,
        steps=params.steps
    )
    time_up_result = calculate_binomial(time_up_params)
    call_theta = (time_up_result.callPrice - base_result.callPrice) / time_change
    put_theta = (time_up_result.putPrice - base_result.putPrice) / time_change
    
    # Vega
    vol_up_params = BinomialParams(
        spotPrice=params.spotPrice,
        strikePrice=params.strikePrice,
        timeToMaturity=params.timeToMaturity,
        volatility=params.volatility + vol_change,
        riskFreeRate=params.riskFreeRate,
        steps=params.steps
    )
    vol_up_result = calculate_binomial(vol_up_params)
    vega = (vol_up_result.callPrice - base_result.callPrice) / vol_change
    
    return {
        "delta": {"call": call_delta, "put": put_delta},
        "gamma": gamma,
        "theta": {"call": call_theta, "put": put_theta},
        "vega": vega,
        "rho": {"call": 0.0, "put": 0.0}  # Rho not calculated for binomial
    }

@app.post("/api/binomial", response_model=BinomialResult)
async def calculate_binomial_endpoint(params: BinomialParams) -> BinomialResult:
    """Calculate Binomial option pricing"""
    return calculate_binomial(params)

@app.post("/api/binomial/greeks-curve", response_model=GreeksCurveResponse)
async def binomial_greeks_curve_endpoint(request: GreeksCurveRequest) -> GreeksCurveResponse:
    """Generate Greeks curve data for Binomial model"""
    x_values: List[float] = []
    y_values: List[float] = []
    
    step = (request.rangeMax - request.rangeMin) / request.steps
    
    for i in range(request.steps + 1):
        if request.curveType == "gamma" or request.curveType == "vega":
            spot_price = request.rangeMin + i * step
            params = BinomialParams(
                spotPrice=spot_price,
                strikePrice=request.strikePrice,
                timeToMaturity=request.timeToMaturity,
                volatility=request.volatility,
                riskFreeRate=request.riskFreeRate,
                steps=50  # Default steps
            )
            x_values.append(spot_price)
        elif request.curveType == "theta":
            time = request.rangeMin + i * step
            params = BinomialParams(
                spotPrice=request.spotPrice,
                strikePrice=request.strikePrice,
                timeToMaturity=time,
                volatility=request.volatility,
                riskFreeRate=request.riskFreeRate,
                steps=50
            )
            x_values.append(time)
        else:
            continue
        
        greeks = calculate_binomial_greeks(params)
        if request.curveType == "gamma":
            y_values.append(greeks["gamma"])
        elif request.curveType == "vega":
            y_values.append(greeks["vega"])
        elif request.curveType == "theta":
            y_values.append(greeks["theta"]["call"])
    
    return GreeksCurveResponse(xValues=x_values, yValues=y_values)

@app.post("/api/monte-carlo", response_model=MonteCarloResult)
async def calculate_monte_carlo_endpoint(params: MonteCarloParams) -> MonteCarloResult:
    """Calculate option prices using Monte Carlo simulation"""
    return calculate_monte_carlo(params)

@app.post("/api/surface-3d", response_model=Surface3DResponse)
async def surface_3d_endpoint(request: Surface3DRequest) -> Surface3DResponse:
    """Generate 3D surface data for option prices"""
    spot_prices: List[float] = []
    times: List[float] = []
    call_prices_grid: List[List[float]] = []
    put_prices_grid: List[List[float]] = []
    
    spot_step = (request.spotMax - request.spotMin) / request.spotSteps
    time_step = (request.timeMax - request.timeMin) / request.timeSteps
    
    for i in range(request.spotSteps + 1):
        spot_price = request.spotMin + i * spot_step
        spot_prices.append(spot_price)
        call_row: List[float] = []
        put_row: List[float] = []
        
        for j in range(request.timeSteps + 1):
            time = request.timeMin + j * time_step
            if i == 0:
                times.append(time)
            
            if request.model == "black-scholes":
                params = BlackScholesParams(
                    spotPrice=spot_price,
                    strikePrice=request.strikePrice,
                    timeToMaturity=time,
                    volatility=request.volatility,
                    riskFreeRate=request.riskFreeRate
                )
                result = calculate_black_scholes(params)
                call_row.append(result.callPrice)
                put_row.append(result.putPrice)
            elif request.model == "binomial":
                params = BinomialParams(
                    spotPrice=spot_price,
                    strikePrice=request.strikePrice,
                    timeToMaturity=time,
                    volatility=request.volatility,
                    riskFreeRate=request.riskFreeRate,
                    steps=request.steps
                )
                result = calculate_binomial(params)
                call_row.append(result.callPrice)
                put_row.append(result.putPrice)
            elif request.model == "monte-carlo":
                params = MonteCarloParams(
                    spotPrice=spot_price,
                    strikePrice=request.strikePrice,
                    timeToMaturity=time,
                    volatility=request.volatility,
                    riskFreeRate=request.riskFreeRate,
                    simulations=request.simulations
                )
                result = calculate_monte_carlo(params)
                call_row.append(result.callPrice)
                put_row.append(result.putPrice)
        
        call_prices_grid.append(call_row)
        put_prices_grid.append(put_row)
    
    return Surface3DResponse(
        spotPrices=spot_prices,
        times=times,
        callPrices=call_prices_grid,
        putPrices=put_prices_grid
    )