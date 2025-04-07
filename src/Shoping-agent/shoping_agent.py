from crewai import Agent, Task
from crewai import Crew, Process
from crewai.tools import tool
from crewai.memory import ShortTermMemory
from crewai.memory.storage.rag_storage import RAGStorage
from typing import List, Optional
import os

gemini_api_key = os.environ["GEMINI_API_KEY"]

products = [
    {"name": "iPhone 11", "price": 700, "quantity": 5},
    {"name": "iPhone 12", "price": 800, "quantity": 10},
    {"name": "iPhone 13", "price": 900, "quantity": 15},
    {"name": "Samsung S21", "price": 1000, "quantity": 20},
    {"name": "Samsung S22", "price": 1100, "quantity": 25},
    {"name": "Samsung S23", "price": 1200, "quantity": 30},
    {"name": "Google Pixel 4", "price": 600, "quantity": 10},
    {"name": "Google Pixel 5", "price": 700, "quantity": 15},
    {"name": "Google Pixel 6", "price": 800, "quantity": 20},
    {"name": "OnePlus 8", "price": 800, "quantity": 15},
    {"name": "OnePlus 9", "price": 900, "quantity": 20},
    {"name": "OnePlus 10", "price": 1000, "quantity": 25},
    {"name": "Huawei P30", "price": 700, "quantity": 10},
    {"name": "Huawei P40", "price": 800, "quantity": 15},
    {"name": "Huawei P50", "price": 900, "quantity": 20},
    {"name": "Oppo Reno 4", "price": 600, "quantity": 10},
    {"name": "Oppo Reno 5", "price": 700, "quantity": 15},
    {"name": "Oppo Reno 6", "price": 800, "quantity": 20},
    {"name": "Xiaomi Redmi 9", "price": 300, "quantity": 10},
    {"name": "Xiaomi Redmi 10", "price": 400, "quantity": 15},
    {"name": "Xiaomi Redmi 11", "price": 500, "quantity": 20},
    {"name": "Sony Xperia 1 III", "price": 1200, "quantity": 10},
    {"name": "Sony Xperia 5 III", "price": 1000, "quantity": 15},
    {"name": "Sony Xperia 10 III", "price": 800, "quantity": 20},
    {"name": "Motorola Moto G", "price": 200, "quantity": 10},
    {"name": "Motorola Moto G Power", "price": 300, "quantity": 15},
    {"name": "Motorola Moto G Stylus", "price": 400, "quantity": 20},
    {"name": "Nokia 5.3", "price": 300, "quantity": 10},
    {"name": "Nokia 7.2", "price": 400, "quantity": 15},
    {"name": "Nokia 8.3", "price": 500, "quantity": 20},
    {"name": "HTC Desire 20 Pro", "price": 400, "quantity": 10},
    {"name": "HTC Desire 21 Pro", "price": 500, "quantity": 15},
    {"name": "HTC Desire 22 Pro", "price": 600, "quantity": 20},
    {"name": "LG G8X ThinQ", "price": 700, "quantity": 10},
    {"name": "LG G9 ThinQ", "price": 800, "quantity": 15},
    {"name": "LG Velvet", "price": 900, "quantity": 20},
]

cart = []

@tool("Find Product")
def find_product(name: str) -> str:
    """Finds and displays details of a specific product."""
    for product in products:
        if product["name"].lower() == name.lower():
            return f"{product['name']} - ${product['price']} (Qty: {product['quantity']})"
    return f"Sorry, {name} is not available in the store."

@tool("Add to Cart")
def add_to_cart(name: str, quantity: int) -> str:
    """Adds a product to the cart and updates the product quantity."""
    for product in products:
        if product["name"].lower() == name.lower():
            if product["quantity"] >= quantity:
                cart.append({"name": product["name"], "price": product["price"], "quantity": quantity})
                product["quantity"] -= quantity
                return f"{name} added to your cart!"
            else:
                return f"Sorry, {name} is out of stock."
    return f"Sorry, {name} is not available in the store."

@tool("Remove from Cart")
def remove_from_cart(name: str, quantity: Optional[int] = None) -> str:
    """Removes a product from the cart and updates the product quantity."""
    for i in range(len(cart)):
        if cart[i]["name"].lower() == name.lower():
            if quantity is None or cart[i]["quantity"] == quantity:
                for p in products:
                    if p["name"].lower() == name.lower():
                        p["quantity"] += cart[i]["quantity"]
                del cart[i]
                return f"{name} has been removed from your cart!"
            if cart[i]["quantity"] < quantity:
                return f"Sorry, you can only remove up to {cart[i]['quantity']} {name} from your cart."
            for p in products:
                if p["name"].lower() == name.lower():
                    p["quantity"] += quantity
            cart[i]["quantity"] -= quantity
            return f"{quantity} {name} has been removed from your cart!"
    return f"Sorry, {name} is not in your cart."

@tool("Show Cart")
def show_cart() -> str:
    """Displays the products in the user's cart."""
    if not cart:
        return "Your cart is empty."
    return "\n".join([f"{p['name']} - ${p['price']} (Qty: {p['quantity']})" for p in cart])

# Disable caching for the show_cart tool
show_cart.cache_function = lambda *args, **kwargs: False
    
@tool("Show All Products")
def show_all_products() -> str:
    """Displays all products in the store."""
    return "\n".join([f"{p['name']} - ${p['price']} (Qty: {p['quantity']})" for p in products])

shopping_assistant = Agent(
    name="Shopping Assistant",
    role="A Shopping Assistant agent",
    goal="To help users shop efficiently by providing product details, managing a cart, and ensuring a smooth shopping experience.",
    backstory="An AI assistant designed to help users shop efficiently by providing product details, managing a cart, and ensuring a smooth shopping experience.",
    tools=[find_product, add_to_cart, remove_from_cart, show_cart, show_all_products ]
)

product_task = Task(
    description="Chat normally with the user maintain a conversation. Helps users find product details and manage their shopping cart efficiently based on their queries. Decides whether they want to view all products, find a product, add it to the cart, view the cart, or continue the conversation. User querries : {user_input}. ",
    agent=shopping_assistant,
    expected_output="Chat normally with the user maintain a conversation. Or returns product details, shopping cart updates, or relevant messages based on user input."
)



embedder_config = {
    "provider": "google",
    "config": {
        "api_key": gemini_api_key,
        "model": "models/text-embedding-004"
    }
}

crew = Crew(
    agents=[shopping_assistant],
    tasks=[product_task],
    process=Process.sequential,
    memory=True,
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config=embedder_config,
            type="short_term",
            path=os.path.join(os.getcwd(), "short_term_storage.db").replace("\\", "/"),
       )
    ),
    embedder=embedder_config,  # Added this line to specify the custom embedder
    # verbose=True,
)

def main():
 while True:
            user_input = input("\033[92mYou: \033[0m") 
            if user_input.lower() in ["exit", "quit"]:
                print("\033[94mBot:\033[0m Goodbye! Have a great day!")  
                break
            response = crew.kickoff(inputs={"user_input": user_input})
            print("\033[94mBot: \033[0m" , response) 


if __name__ == "__main__":
    main()