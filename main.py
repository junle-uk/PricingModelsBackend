from typing import Union
from typing import Dict
import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard normal cumulative distribution function
def normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF using Abramowitz and Stegun formula"""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = -1 if x < 0 else 1
    x = abs(x) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)

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
    call_price = params.spotPrice * normal_cdf(d1) - params.strikePrice * discount_factor * normal_cdf(d2)
    put_price = params.strikePrice * discount_factor * normal_cdf(-d2) - params.spotPrice * normal_cdf(-d1)
    
    return BlackScholesResult(
        callPrice=max(0.0, call_price),
        putPrice=max(0.0, put_price),
        d1=d1,
        d2=d2
    )

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None) -> Dict[str,Union[str, None]]:
    return {"item_id": item_id, "q": q}

@app.post("/api/black-scholes", response_model=BlackScholesResult)
async def calculate_black_scholes_endpoint(params: BlackScholesParams) -> BlackScholesResult:
    """Calculate Black-Scholes option pricing"""
    return calculate_black_scholes(params)