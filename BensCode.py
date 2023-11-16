#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

#dataframe_to_dictionary is a method from Gabriel Gullidge (student number 220328397). Used with permission
#stores the data as managable dictionary and arrays instead of pandas dataframe object
def dataframe_to_dictionary(df):
    dictionary = {}
    for header in df:
        dictionary[header] = [data for data in df[header]]
    return dictionary

##Video_games print

df = pd.read_excel("statistic_id274072_best-selling-video-games-in-the-uk-2021.xlsx", "Data", header = 2)
df.head()
print(df)

x = dataframe_to_dictionary(df)
best_selling_games1 = x["Best-selling video games in the UK 2021"]

best_selling_games = np.array(best_selling_games1)
print(best_selling_games)

x = dataframe_to_dictionary(df)
best_selling_sales1 = x["Sales"]

best_selling_sales = np.array(best_selling_sales1)
print(best_selling_sales)

fig = plt.figure(figsize = (10, 5))

##video games bar chart

# creating the bar plot
plt.bar(best_selling_games, best_selling_sales, color = 'red',
        width = 0.4)
 
plt.xlabel("Games (Top 20 UK sold in 2021)")
plt.ylabel("Number of UK sales in 2021")
plt.xticks(rotation=90)
plt.title("UK video game sales in 2021")
plt.show()

##video games print

df = pd.read_excel("statistic_id274072_best-selling-video-games-in-the-uk-2021.xlsx", "Data.3", header = 2)
df.head()
print(df)

x = dataframe_to_dictionary(df)
cert_18_percentage1 = x["Cert 18 percentage"]

cert_18_percentage = np.array(cert_18_percentage1)
print(cert_18_percentage)

x = dataframe_to_dictionary(df)
not_cert_18_percentage1 = x["not Cert 18 without fifa 22 percentage"]

not_cert_18_percentage = np.array(not_cert_18_percentage1)
print(not_cert_18_percentage)

x = dataframe_to_dictionary(df)
Fifa_221 = x["Fifa 22 Percentage"]

Fifa_22 = np.array(Fifa_221)
print(Fifa_22)

x = dataframe_to_dictionary(df)
total_percentage_games1 = x["Total percentage"]

total_percentage_games = np.array(total_percentage_games1)
print(total_percentage_games)

Type_of_games = ['Certficate 18 (4185225)', 'Fifa 22 (2338778)', 'Not certificate 18 excluding fifa 22 (4228941)']
 
data = [cert_18_percentage, Fifa_22, not_cert_18_percentage]
 
##video games pie chart
# Create a pie chart using Plotly Express
#Type_of_games = ['Certficate 18 (4185225)', 'Fifa 22 (2338778)', 'Not certificate 18 excluding fifa 22 (4228941)']
 
#data = [cert_18_percentage, Fifa_22, not_cert_18_percentage]
 
# Creating plot
#fig = plt.figure(figsize =(10, 7))
#plt.pie(data, labels = Type_of_games)
#plt.title("Pie chart to show top 20 UK games sold by certificate")
#plt.legend(loc = 'upper left')
 
# show plot
#plt.show()

##more gaming data

df = pd.read_excel("statistic_id274072_best-selling-video-games-in-the-uk-2021.xlsx", "Data2", header = 2)
df.head()
print(df)

x = dataframe_to_dictionary(df)
game_genre_use1= x["Use"]

game_genre_use = np.array(game_genre_use1)
print(game_genre_use)

x = dataframe_to_dictionary(df)
game_genre_shooter1 = x["Shooter gamers"]

game_genre_shooter = np.array(game_genre_shooter1)
print(game_genre_shooter)

x = dataframe_to_dictionary(df)
game_genre_all1 = x["All genre gamers"]

game_genre_all = np.array(game_genre_all1)
print(game_genre_all)

##gaming data bar chart 

X = ['Heavy','Medium','Light','Ultra light', 'Do not know']

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, game_genre_shooter, 0.4, label = 'Shooter genre gamers')
plt.bar(X_axis + 0.2, game_genre_all, 0.4, label = 'All genre gamers')

plt.xticks(X_axis, X)
plt.xlabel("Genre extent of use")
plt.ylabel("Percentage of adult gamers")
plt.title("Comparision between all and shooter adult gaming genres in the UK")
plt.legend()
plt.show()

##more gaming data

df = pd.read_excel("statistic_id274072_best-selling-video-games-in-the-uk-2021.xlsx", "Data3", header = 2)
df.head()
print(df)

x = dataframe_to_dictionary(df)
game_genre = x["Genre "]
print(game_genre)

x = dataframe_to_dictionary(df)
gane_genre_percentage = x["Percentage"]
print(gane_genre_percentage)

##crime data

df = pd.read_excel("statistic_id288256_number-of-violent-crime-offences-in-england-and-wales-2007-2022.xlsx", "Data", header = 3)
df.head()
print(df)

x = dataframe_to_dictionary(df)
violent_crimes1 = x["Crimes"]

z = dataframe_to_dictionary(df)
data_dates1 = z["Date"]

violent_crimes = np.array(violent_crimes1)
print(violent_crimes)

data_dates = np.array(data_dates1)
print(data_dates)

##gamers data 

df = pd.read_excel("statistic_id300521_uk-gaming-reach-2007-2021.xlsx", "Data", header = 2)
df.head()
print(df)

y = dataframe_to_dictionary(df)
video_gamer1 = y["Gamers"]

z = dataframe_to_dictionary(df)
data_dates1 = z["UK gaming reach 2007-2021"]

video_gamer = np.array(video_gamer1)
print(video_gamer)

data_dates = np.array(data_dates1)
print(data_dates)

##sexual offences data

df = pd.read_excel("Figure_10__Suspects_convicted_of_homicide_show_a_younger_age_profile (2).xlsx", "Data5", header = 2)
df.head()
print(df)

y = dataframe_to_dictionary(df)
data_dates = y["Year"]

y = dataframe_to_dictionary(df)
sexual_offences1 = y["Sexual Offences"]

data_dates = np.array(data_dates1)
print(data_dates)

sexual_offences = np.array(sexual_offences1)
print(sexual_offences)

##wepon possession data

df = pd.read_excel("statistic_id828895_number-of-possession-of-weapon-offences-england-and-wales-2003-2022 (1) (1).xlsx", "Data", header = 2)
df.head()
print(df)

y = dataframe_to_dictionary(df)
weapon_possession1 = y["weapon possession"]

z = dataframe_to_dictionary(df)
data_dates = z["Year"]

data_dates = np.array(data_dates1)
print(data_dates)

weapon_possession = np.array(weapon_possession1)
print(weapon_possession)

##suicide rate 

df = pd.read_excel("statistic_id282160_suicide-rate-in-england-and-wales-2000-2021 (1).xlsx", "Data", header = 3)
df.head()
print(df)

x = dataframe_to_dictionary(df)
data_dates1 = x["Date"]

y = dataframe_to_dictionary(df)
suicide_rate1 = y["Suicide rate "]

data_dates = np.array(data_dates1)
print(data_dates)

suicide_rate = np.array(suicide_rate1)
print(suicide_rate)

##more sexual offences 

df = pd.read_excel("Figure_10__Suspects_convicted_of_homicide_show_a_younger_age_profile (2).xlsx", "Data4", header = 2)
df.head()
print(df)

y = dataframe_to_dictionary(df)
smaller_year = y["Year"]
print(smaller_year)

y = dataframe_to_dictionary(df)
sexual_offences_small = y["Sexual Offences"]
print(sexual_offences_small)

##more gamer data 

df = pd.read_excel("statistic_id300513_uk-gaming-reach-2013-2021-by-age-group-and-gender (1) (1) (2).xlsx", "Data3", header = 2)
df.head()
print(df)

z = dataframe_to_dictionary(df)
smaller_year1 = z["Year"]

smaller_year = np.array(smaller_year1)
print(smaller_year)

z = dataframe_to_dictionary(df)
gamer_age_16 = z["16-24"]

gamer_age_161 = np.array(gamer_age_16)
print(gamer_age_161)

z = dataframe_to_dictionary(df)
gamer_age_25 = z["25-34"]

gamer_age_251 = np.array(gamer_age_25)
print(gamer_age_251)

z = dataframe_to_dictionary(df)
gamer_age_35 = z["35-44"]

gamer_age_351 = np.array(gamer_age_35)
print(gamer_age_351)

z = dataframe_to_dictionary(df)
gamer_age_45 = z["45-54"]

gamer_age_451 = np.array(gamer_age_45)
print(gamer_age_451)

z = dataframe_to_dictionary(df)
gamer_age_55 = z["55-64"]

gamer_age_551 = np.array(gamer_age_55)
print(gamer_age_551)

z = dataframe_to_dictionary(df)
gamer_age_65 = z["65-74"]

gamer_age_651 = np.array(gamer_age_65)
print(gamer_age_651)

##gammer multi bar chart

X = ['2017', '2018', '2019', '2020', '2021']

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, gamer_age_161, 0.4, label = '16-24')
plt.bar(X_axis + 0.2, gamer_age_251, 0.4, label = '25-34')
plt.bar(X_axis - 0.2, gamer_age_351, 0.4, label = '35-44')
plt.bar(X_axis + 0.2, gamer_age_451, 0.4, label = '45-54')
plt.bar(X_axis - 0.2, gamer_age_551, 0.4, label = '55-64')
plt.bar(X_axis + 0.2, gamer_age_651, 0.4, label = '65-74')

plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Percentage of gamers in the UK")
plt.title("UK video gamers yearly by age group")
plt.legend()
plt.show()

##another multi bar chart for gamer data

import plotly.graph_objects as px

 
# creating random data through randomint
# function of numpy.random
np.random.seed(42)
 
random_x= np.random.randint(1,101,100)
random_y= np.random.randint(1,101,100)
 
x = smaller_year
 
plot = px.Figure(data=[px.Bar(
    name = '16-24',
    x = x,
    y = gamer_age_161
   ),
                       px.Bar(
    name = '25-34',
    x = x,
    y = gamer_age_251
   ),                  px.Bar(
    name = '35-44',
    x = x,
    y = gamer_age_351
   ),                  px.Bar(
    name = '45-54',
    x = x,
    y = gamer_age_451
   ),                  px.Bar(
    name = '55-64',
    x = x,
    y = gamer_age_551
   ),                  px.Bar(
    name = '65-74',
    x = x,
    y = gamer_age_651
   )
])
plot.show()

##homocide data 

df = pd.read_excel("Figure_10__Suspects_convicted_of_homicide_show_a_younger_age_profile (2).xlsx", "Data", header = 2)
df.head()
print(df)

z = dataframe_to_dictionary(df)
smaller_year1 = z["Year"]

smaller_year = np.array(smaller_year1)
print(smaller_year)

z = dataframe_to_dictionary(df)
homicide_age_16 = z["16-24"]

homicide_age_161 = np.array(homicide_age_16)
print(homicide_age_161)

z = dataframe_to_dictionary(df)
homicide_age_25 = z["25-34"]

homicide_age_251 = np.array(homicide_age_25)
print(homicide_age_251)

z = dataframe_to_dictionary(df)
homicide_age_35 = z["35-44"]

homicide_age_351 = np.array(homicide_age_35)
print(homicide_age_351)

z = dataframe_to_dictionary(df)
homicide_age_45 = z["45-54"]

homicide_age_451 = np.array(homicide_age_45)
print(homicide_age_451)

z = dataframe_to_dictionary(df)
homicide_age_55 = z["55-64"]

homicide_age_551 = np.array(homicide_age_55)
print(homicide_age_551)

z = dataframe_to_dictionary(df)
homicide_age_65 = z["65-74"]

homicide_age_651 = np.array(homicide_age_65)
print(homicide_age_651)

##homicide multi bar chart

X = smaller_year

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.5, homicide_age_161, 0.2, label = '16-24')
plt.bar(X_axis - 0.3, homicide_age_251, 0.2, label = '25-34')
plt.bar(X_axis - 0.1, homicide_age_351, 0.2, label = '35-44')
plt.bar(X_axis + 0.1, homicide_age_451, 0.2, label = '45-54')
plt.bar(X_axis + 0.3, homicide_age_551, 0.2, label = '55-64')
plt.bar(X_axis + 0.5, homicide_age_651, 0.2, label = '65-74')

plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Percentage of gamers in the UK")
plt.title("UK video gamers yearly by age group")
plt.legend(loc = 'right')
plt.show()

##homicide another multi bar chart

import plotly.graph_objects as px

# creating random data through randomint
# function of numpy.random
np.random.seed(42)
 
random_x= np.random.randint(1,101,100)
random_y= np.random.randint(1,101,100)
 
x = smaller_year
 
plot = px.Figure(data=[px.Bar(
    name = '16-24',
    x = x,
    y = homicide_age_161
   ),
                       px.Bar(
    name = '25-34',
    x = x,
    y = homicide_age_251
   ),                  px.Bar(
    name = '35-44',
    x = x,
    y = homicide_age_351
   ),                  px.Bar(
    name = '45-54',
    x = x,
    y = homicide_age_451
   ),                  px.Bar(
    name = '55-64',
    x = x,
    y = homicide_age_551
   ),                  px.Bar(
    name = '65-74',
    x = x,
    y = homicide_age_651
   )
])
plot.show()

##gamer gander data

df = pd.read_excel("statistic_id300513_uk-gaming-reach-2013-2021-by-age-group-and-gender (1) (1) (2).xlsx", "Data2", header = 2)
df.head()
print(df)

z = dataframe_to_dictionary(df)
smaller_year = z["Year"]
print(smaller_year)

z = dataframe_to_dictionary(df)
male_gamer = z["Male"]
print(male_gamer)

z = dataframe_to_dictionary(df)
female_gamer = z["Female"]
print(female_gamer)

##multi bar chat for gamer gender

X = ['2017', '2018', '2019', '2020', '2021']

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, male_gamer, 0.4, label = 'Male gamer')
plt.bar(X_axis + 0.2, female_gamer, 0.4, label = 'Female gamer')

plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Percentage of gamers in the UK")
plt.title("UK video gamers by adult gender from 2017-2021")
plt.legend(loc = 'best')
plt.show()

##homicide data for gender

df = pd.read_excel("Figure_10__Suspects_convicted_of_homicide_show_a_younger_age_profile (2).xlsx", "Data2", header = 2)
df.head()
print(df)

z = dataframe_to_dictionary(df)
smaller_year = z["Year"]
print(smaller_year)

z = dataframe_to_dictionary(df)
male_homicide = z["Male"]
print(male_homicide)

z = dataframe_to_dictionary(df)
female_homicide = z["Female"]
print(female_homicide)

##homicide by gender multi bar chart

X = ['2017', '2018', '2019', '2020', '2021']

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, male_homicide, 0.4, label = 'Male homicider')
plt.bar(X_axis + 0.2, female_homicide, 0.4, label = 'Female homicider')

plt.xticks(X_axis, X)
plt.xlabel("Years")
plt.ylabel("Number of homicide offences")
plt.title("Adult homicide offences yearly by gender in the UK")
plt.legend(loc = 'best')
plt.show()

##suicide rate

df = pd.read_excel("statistic_id289102_suicide-rate-in-england-and-wales-2021-by-age (1).xlsx", "Data", header = 2)
df.head()
print(df)

z = dataframe_to_dictionary(df)
age_group = z["Age"]
print(age_group)

z = dataframe_to_dictionary(df)
suicide_rate_age = z["Suicide rate"]
print(suicide_rate_age)

##suicide rate bar chart

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(age_group, suicide_rate_age, color = 'blue',
        width = 0.4)
 
plt.xlabel("Age group")
plt.ylabel("Adult suicide rate")
plt.xticks(rotation=90)
plt.title("UK adult suicide rate in 2021 by age group")
plt.show()

##more homicdies data

df = pd.read_excel("statistic_id283093_number-of-homicides-in-england-and-wales-2002-2022.xlsx", "Data", header = 1)
df.head()
print(df)

z = dataframe_to_dictionary(df)
twelve_dates1 = z["Year"]

z = dataframe_to_dictionary(df)
homicide_rate1 = z["Number of homicides in England and Wales 2002-2022"]

twelve_dates = np.array(twelve_dates1)
print(twelve_dates)

homicide_rate = np.array(homicide_rate1)
print(homicide_rate)

##more gamers data

df = pd.read_excel("statistic_id283093_number-of-homicides-in-england-and-wales-2002-2022.xlsx", "Data2", header = 2)
df.head()
print(df)

z = dataframe_to_dictionary(df)
twelve_dates1 = z["Year"]

z = dataframe_to_dictionary(df)
gamingtwelve_rate1 = z["Games"]

twelve_dates = np.array(twelve_dates1)
print(twelve_dates)

gamingtwelve_rate = np.array(gamingtwelve_rate1)
print(gamingtwelve_rate)

##3D scatter graph for homicde age and gamer age

import plotly.express as px

ages = ["16-24","16-24","16-24","16-24","16-24", "25-34","25-34","25-34","25-34","25-34","35-44","35-44","35-44","35-44","35-44", "45-54","45-54","45-54","45-54","45-54","55-64","55-64","55-64","55-64","55-64", "65-74","65-74","65-74","65-74","65-74"]

gamers = gamer_age_16+ gamer_age_25+ gamer_age_35+ gamer_age_45+ gamer_age_55+ gamer_age_65

homiciders = homicide_age_16+homicide_age_25+ homicide_age_35+ homicide_age_45+ homicide_age_55+ homicide_age_65
newDict = {"Age Group":ages, "Gaming":gamers, "Homicides":homiciders}
#print(newDict)
data = pd.DataFrame(newDict)
#print(data["Age Group"])
fig = px.scatter_3d(data, x="Age Group", y="Gaming", z="Homicides", color="Age Group", width=1000, title = "3D scatter plot showing UK adult video gaming against homicides by age group from 2017-2022")
fig.update_traces(marker_size=5)
fig.show()

##coreelation 

np.corrcoef(video_gamer,violent_crimes)[0,1]

np.corrcoef(video_gamer,sexual_offences)[0,1]

np.corrcoef(video_gamer,weapon_possession)[0,1]

np.corrcoef(gamingtwelve_rate,homicide_rate)[0,1]

np.corrcoef(video_gamer,suicide_rate)[0,1]