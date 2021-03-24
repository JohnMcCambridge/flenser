# Flenser

*Have you ever been handed a dataset you've never seen before?*

Flenser is a simple, minimal, automated exploratory data analysis tool. It runs a set of simple tests against each column within a dataset, 
and outputs a HTML file noting which tests trigger per column, alongside relevant outputs.

Flenser is intended to be run at the earliest stages of data exploration, when you have no familiarity with the dataset. 
It will do its best to tell you what is actually going on in the dataset, regardless of what is *supposed* to be going on in the dataset.

Flenser is designed to be helpful, not 'helpful': it will not attempt to modify or make assumptions about your dataset. Instead it will apply each simple test, 
to every column, and show you outputs that will allow your human brain to make decisions about what is actually going on.

Additional tests can be added by modifying the `Test` dataclass. 

### How to run 
python3 flenser.py "*filename.csv*"

### With thanks to

Recurse <br>
Kelly F <br>
Rebecca H <br>
Azhad S <br>
Shivam S <br>
Christina M <br>
Adam K <br>
Edith V <br>
Justin R <br>
