import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.stats import norm

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

@app.post("/api/black-scholes", response_model=BlackScholesResult)
async def calculate_black_scholes_endpoint(params: BlackScholesParams) -> BlackScholesResult:
    """Calculate Black-Scholes option pricing"""
    return calculate_black_scholes(params)