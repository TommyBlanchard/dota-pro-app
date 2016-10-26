from flask import Flask, render_template, request, redirect
import sys
import logging
import dill
import pandas as pd
import itertools
from scipy.sparse import hstack
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, Range1d
from bokeh.embed import components
from bokeh.layouts import row

app = Flask(__name__)

app.vars={}

app.vars['model'] = dill.load(open( "model.p", "rb" ) )
app.vars['sorted_heroes'] = dill.load(open('sorted_heroes.p', 'rb'))
app.vars['d_sorted_heroes'] = dict(dill.load(open('sorted_heroes.p', 'rb')))
app.vars['num_dict'] = dict([(num, name) for (name,num) in app.vars['sorted_heroes']])
app.vars['combos_vectorizer'] = dill.load(open( "combos_vectorizer.p", "rb" ))
app.vars['hero_vectorizer'] = dill.load(open( "hero_vectorizer.p", "rb" ))
    
@app.route('/')
def main():
    return redirect('/index')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/heroes')
def heroes():
    return render_template('heroes.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')


@app.route('/predictin')
def predictin():

    hero_str = ''
    for hero in app.vars['sorted_heroes']:
        hero_str += '      <option data-value="' + hero[1] + '" value="' + hero[0] + '">' + hero[0] + '</option>\n'

    html = '<form id="input" method="post" action="predictout"><table><tr>\n    <th>Radiant Heroes</th><th>Radiant Lanes</th>\n    <td>&nbsp;&nbsp;&nbsp;&nbsp;</td><th>Dire Heroes</th><th>Dire Lanes</th>\n</tr>'

    for i in range(5):
        html+='<tr><td><input type="text" name="rad' + str(i) + '" list="rad' + str(i) + '">\n  <datalist id="rad' + str(i) + '">\n    <select>\n'
        html += hero_str
        html += '    </select>\n  </datalist></td>'
        html += '  <td><select name="radlane' + str(i) + '">\n'
        html += '<option value="1">Bottom</option>'
        html += '<option value="2">Middle</option>'
        html += '<option value="3">Top</option>'
        html += '<option value="4">Radiant Jungle</option>'
        html += '<option value="5">Dire Jungle</option>'
        html += '</select></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td>'

        html+='<td><input type="text" name="dire' + str(i) + '"list="dire' + str(i) + '">\n  <datalist id="dire' + str(i) + '">\n    <select>\n'
        html += hero_str
        html += '    </select>\n  </datalist></td>'
        html += '  <td><select name="direlane' + str(i) + '">\n'
        html += '<option value="1">Bottom</option>'
        html += '<option value="2">Middle</option>'
        html += '<option value="3">Top</option>'
        html += '<option value="4">Radiant Jungle</option>'
        html += '<option value="5">Dire Jungle</option>'
        html += '</select></td></tr>'
    
    html += '</table><input type="submit" value="Submit"></form>'
    
    return render_template('predictin.html', script=html)

@app.route('/predictout', methods=['POST'])
def predictout():
    if request.method == 'POST':
        rad = []
        dire = []
        radlane = []
        direlane = []
        for i in range(5):
            rad.append(str(app.vars['d_sorted_heroes'][request.form['rad' + str(i)]]))
            radlane.append(request.form['radlane' + str(i)])
            dire.append(str(app.vars['d_sorted_heroes'][request.form['dire' + str(i)]]))
            direlane.append(request.form['direlane' + str(i)])

        X = pd.DataFrame({ 'rad_heroes': [rad], 'dire_heroes': [dire]})
        
        rad_heroes = app.vars['hero_vectorizer'].transform(X['rad_heroes'])
        dire_heroes = app.vars['hero_vectorizer'].transform(X['dire_heroes'])
        her = rad_heroes - dire_heroes

        rad_com = app.vars['combos_vectorizer'].transform(
            list(X['rad_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2)))))
        dire_com = app.vars['combos_vectorizer'].transform(
            list(X['dire_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2)))))
        com = rad_com - dire_com
        x_in = hstack((her,com))
        prob = app.vars['model'].predict_proba(x_in)

        script = "<center><h3>Probability of Radiant winning: %.1f%%</h3>" % (prob[0][1]*100)        

        ##Make plot
        rad_combos = list(X['rad_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2))))
        dire_combos = list(X['dire_heroes'].apply(lambda x: list(itertools.combinations(sorted(x),2))))

        name = []
        val = []
        for x in rad:
            try:
                val.append(app.vars['model'].coef_[0]
                           [app.vars['hero_vectorizer'].transform(
                            pd.Series([[str(x)]])).nonzero()[1]][0])
                name.append('Radiant ' + app.vars['num_dict'][x])
            except:
                pass

        for x in dire:
            try:
                val.append(-app.vars['model'].coef_[0]
                           [app.vars['hero_vectorizer'].transform(
                            pd.Series([[str(x)]])).nonzero()[1]][0])   
                name.append('Dire ' + app.vars['num_dict'][x])
            except:
                pass

        for x in rad_combos[0]:
            try:
                val.append(app.vars['model'].coef_[0][hstack((
                                app.vars['hero_vectorizer'].transform('n'),
                                app.vars['combos_vectorizer'].transform(pd.Series([[x]])))).nonzero()[1][0]])
                name.append('Radiant ' + app.vars['num_dict'][x[0]] + ' and ' + app.vars['num_dict'][x[1]])
            except:
                pass

        for x in dire_combos[0]:
            try:
                val.append(-app.vars['model'].coef_[0][hstack((
                                app.vars['hero_vectorizer'].transform('n'),
                                app.vars['combos_vectorizer'].transform(pd.Series([[x]])))).nonzero()[1][0]])
                name.append('Dire ' + app.vars['num_dict'][x[0]] + ' and ' + app.vars['num_dict'][x[1]])
            except:
                pass
            

        dat = zip(val,name)

        ylim = max([abs(item) for item in val])

        pos_dat = sorted([(val, name) for (val,name) in dat if val > 0], reverse=True)
        neg_dat = sorted([(val, name) for (val,name) in dat if val < 0])

        pos_vals = [val for (val, name) in pos_dat]
        pos_names = [name for (val, name) in pos_dat]

        neg_vals = [val for (val, name) in neg_dat]
        neg_names = [name for (val, name) in neg_dat]

        tooltips=[
            ('Factor', '@desc'),
            ('Value', '@vals')
        ]

        p1 = figure(plot_width=500, plot_height=400, tools="", title='Factors Helping Radiant')

        h = pos_vals
        adj_h = [x/2 for x in h]
        print(adj_h)
        source1 = ColumnDataSource(
                data=dict(
                    x=range(len(h)),
                    y=adj_h,
                    height = h,
                    desc=pos_names,
                    vals=pos_vals,
                )
            )

        cr1 = p1.rect(x='x', y='y', width=0.8, alpha=.3, height='height', fill_color="#00FF00",  hover_fill_color="yellow", source=source1)

        p1.add_tools(HoverTool(tooltips=tooltips, renderers=[cr1]))

        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.visible = False
        p1.title.text_color = "green"

        p2 = figure(plot_width=500, plot_height=400, tools="", title='Factors Helping Dire')

        h = [-item for item in neg_vals]
        adj_h = [x/2 for x in h]

        source2 = ColumnDataSource(
                data=dict(
                    x=range(len(h)),
                    y=adj_h,
                    height = h,
                    desc=neg_names,
                    vals=h,
                )
            )

        cr2 = p2.rect(x='x', y='y', width=0.8, alpha=.3, height='height', fill_color="#FF0000",  hover_fill_color="yellow", source=source2)

        p2.add_tools(HoverTool(tooltips=tooltips, renderers=[cr2]))

        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.xaxis.visible = False
        p2.title.text_color = "red"

        p1.set(y_range=Range1d(0, ylim))
        p2.set (y_range=Range1d(0, ylim))
        
        script2, div = components(row(p1,p2))
        return render_template('predictout.html', script=script, script2=script2, div=div)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(port=33507)