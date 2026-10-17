from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os

# 🚀 Setup Selenium (Chrome)
options = webdriver.ChromeOptions()
options.add_argument("--headless")   # run without opening browser
driver = webdriver.Chrome(options=options)

# 🚀 Open Soil Health Portal
driver.get("https://soilhealth.dac.gov.in/nutrient-dashboard")

wait = WebDriverWait(driver, 15)

# -------- Functions --------
def get_soil_data(state_name, district_name):
    # Select State
    state_dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "ddlState"))))
    state_dropdown.select_by_visible_text(state_name)
    time.sleep(2)

    # Select District
    district_dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "ddlDistrict"))))
    district_dropdown.select_by_visible_text(district_name)
    time.sleep(2)

    # Click Submit
    submit_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnSearch")))
    submit_btn.click()
    time.sleep(3)

    # Scrape Table
    table = wait.until(EC.presence_of_element_located((By.ID, "gvSoilData")))
    rows = table.find_elements(By.TAG_NAME, "tr")

    data = []
    for row in rows:
        cols = [col.text for col in row.find_elements(By.TAG_NAME, "td")]
        if cols:
            data.append(cols)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Parameter", "Value"])
    return df


# -------- Example Usage --------
soil_df = get_soil_data("Bihar", "Begusarai")
print(soil_df)

# Save to CSV
soil_df.to_csv("soil_data_begusarai.csv", index=False)

driver.quit()