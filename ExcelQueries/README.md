Query:Smallest city in the largest country in terms of population whose official language is not english

# Prompt

You are working with a postgresql database.
The programmer issues commands and you should translate them into SQL queries.

Human: Smallest city in the largest country in terms of population whose official language is not english

```

CREATE TABLE city
(
    id integer NOT NULL,
    name text NOT NULL,
    countrycode character(3) NOT NULL,
    district text NOT NULL,
    population integer NOT NULL
);
SAMPLE ROW:
(1, 'Kabul', 'AFG', 'Kabol', 1780000)
CREATE TABLE country
(
    code character(3) NOT NULL,
    name text NOT NULL,
    continent text NOT NULL,
    region text NOT NULL,
    surfacearea real NOT NULL,
    indepyear smallint NULL,
    population integer NOT NULL,
    lifeexpectancy real NULL,
    gnp numeric(10,2) NULL,
    gnpold numeric(10,2) NULL,
    localname text NOT NULL,
    governmentform text NOT NULL,
    headofstate text NULL,
    capital integer NULL,
    code2 character(2) NOT NULL
);
SAMPLE ROW:
('AFG', 'Afghanistan', 'Asia', 'Southern and Central Asia', 652090.0, 1919, 22720000, 45.9, Decimal('5976.00'), None, 'Afganistan/Afqanestan', 'Islamic Emirate', 'Mohammad Omar', 1, 'AF')
CREATE TABLE countrylanguage
(
    countrycode character(3) NOT NULL,
    language text NOT NULL,
    isofficial boolean NOT NULL,
    percentage real NOT NULL
);
SAMPLE ROW:
('AFG', 'Pashto', True, 52.4)
Foriegn Keys:
Foreign Key in table country in capital referencing table city column id
Foreign Key in table countrylanguage in countrycode referencing table country column code

```

Reply in the format:
{
"code": string
}

# SQL generated

{
"code": "SELECT city.name FROM city JOIN country ON city.countrycode = country.code JOIN countrylanguage ON
country.code = countrylanguage.countrycode WHERE countrylanguage.isofficial = false AND countrylanguage.language != '
English' ORDER BY country.population DESC, city.population ASC LIMIT 1"
}
# SQL results
[('Huangyan',)]

