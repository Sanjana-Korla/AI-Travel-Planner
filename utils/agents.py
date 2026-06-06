from utils.gemini_helper import get_ai_response


# Trip Planner Agent
def trip_planner_agent(location, days, context):

    prompt = f"""

    Using this information:

    {context}

    Create a short {days}-day travel itinerary for {location}.

    Format in HTML.

    Use:
    <h3> for headings
    <ul><li> for points

    Keep response short and attractive.

    Do not include ```html or ``` in response.

    """

    return get_ai_response(prompt)



# Hotel Recommendation Agent
def hotel_agent(location, budget):

    prompt = f"""

    Suggest 3 good hotels in {location}
    within budget {budget}.

    Include:
    - Hotel name
    - Approximate price
    - Nearby attraction

    Format in HTML.

    Use bullet points only.

    Keep output short.

    Do not include ```html or ``` in response.

    """

    return get_ai_response(prompt)



# Budget Planning Agent
def budget_agent(location, budget, days):

    prompt = f"""

    Create a short budget plan for a {days}-day trip to {location}
    within budget {budget}.
    Keep total budget close to {budget}.

    Include:
    - Hotel
    - Food
    - Transport
    - Activities

    Format in HTML.

    Use bullet points.

    Keep output concise.

    Do not include ```html or ``` in response.

    """

    return get_ai_response(prompt)