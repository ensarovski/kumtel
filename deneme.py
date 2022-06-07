import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 
import pandas as pd
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
       

from ortools.sat.python import cp_model
st.set_page_config(page_title='Kumtel Project', page_icon='😎')
logo = Image.open('kumtel.jpg')
profile = Image.open('kumtel.jpg')

with st.sidebar:
    choose = option_menu("MENU", ["About Project","Scheduling Program",  "Source Code", "Extras", "Contact"],
                         icons=['kanban ', 'bi bi-wrench', 'bi bi-code-slash', 'bi bi-brightness-low','person lines fill'],
                         menu_icon="bi bi-list", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "red", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )


logo = Image.open('kumtel.jpg')
profile = Image.open('kumtel.jpg')


if choose == "About Project":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: teal;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Welcome to the Kumtel Project</p>', unsafe_allow_html=True)      
    image = Image.open('kumtel.jpg')
    st.image(image)
    st.markdown('Kumtel is a company that produces hundreds of products in 6 different categories. These categories are hobs, heaters and coolers, hoods, mini and midi ovens, built in ovens, and free standing ovens.')
    urun = Image.open('urun.jpeg')
    st.image(urun)
    st.markdown('Some of the parts required for the production of these products are produced by Kumtel by supplying raw materials, while some are bought ready-made from the suppliers. The materials taken as raw materials pass through different processes and form the final product. The most important of these processes is the pressing process, because almost all of the parts of any product produced in Kumtel require pressing.')
    st.write('')
    st.markdown('Here the list of machines:')
    makine = Image.open('makine.jpeg')
    st.image(makine)
    st.write('')
    st.write('')
    st.markdown('Here the list of products and their routes:')
    df =pd.read_excel('gurup.xlsx')
    st.write(df.head(20))
    st.write('')
    st.markdown('These molds are attached to suitable machines and press process is performed. However, there are problems in assigning the molds to the machines due to the excess in the number of products and machines and the necessity of ensuring the precedence relationships of the parts.')
    st.write('')
    st.markdown('Thanks to this web application, this problem is solved. The developed web application is run by entering the products to be produced and the demands of these products. In addition, it can be specified on disabled machines. After this data is entered, this web application is run and the result is obtained.')
    st.write('')
    st.markdown('So, why is a web application preferred over a desktop application? In the web application, the processes can be handled only through the browser without requiring manual download or update, which provides flexibility for the employees. It can be used without any difficulties as it does not require a download process as in desktop applications.')
    


elif choose == "Scheduling Program":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: teal;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Scheduling Program</p>', unsafe_allow_html=True)    


    options = st.multiselect(
     'Şuan çalışamayacak durumdaki makineler nelerdir?',
     ['press200-1', 'press160-1', 'press110-1', 'press60-1', 'press25-1', 'press500-1', 'press400-1', 'press160-2', 'press110-2', 'pressŞemmak'])
    
    order_count =st.number_input('Siparişte kaç farklı ürün var?',min_value=0,max_value=250,step=1)
    order=[0]*order_count
    order2=[0]*order_count
    secenek =st.radio("verileri nasıl girmek istersiniz?",('manuel olarak','dosya yükleyerek'))

    if secenek=='manuel olarak':
        for i in range(order_count):
            order[i]=st.selectbox('ürünün kodu ne?',key=i,options=['urun1','urun2','urun3','urun4','urun5','urun6'])
            order2[i]=st.slider('üründen kaç adet sipariş geldi?',key=i,min_value=5,max_value=100,step=1)
        class SolutionPrinter(cp_model.CpSolverSolutionCallback):


            def __init__(self):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.__solution_count = 0

            def on_solution_callback(self):
                """Called at each new solution."""
                print('Solution %i, time = %f s, objective = %i' %
                      (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
                self.__solution_count += 1


        def flexible_jobshop():

            a = 0
            jobs=[]



            urun11=[  # task = (processing_time, machine_id)
                    [(3, 0), (3, 1), (3, 2), (3, 3)],  # task 0 with 3 alternatives
                    [(3, 0), (3, 1), (3, 2), (3, 3)],  # task 1 with 3 alternatives
                    [(3, 0), (3, 1), (3, 2), (3, 3)],
                ]


            urun22=[  # task = (processing_time, machine_id)
                    [(3, 0), (3, 1), (4, 2), (3, 3)],  # task 0 with 3 alternatives
                    [(3, 0), (3, 1), (3, 3)],  # task 1 with 2 alternatives
                    [(3, 0), (3, 2), (3, 3)],
                ]



            urun33=[  # task = (processing_time, machine_id)
                    [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                    [(3, 0), (3, 1)],  # task 1 with 3 alternatives
                    [(3, 2), (3, 1)],
                ]
            
            urun44=[  # task = (processing_time, machine_id)
                    [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                    [(3, 0), (5, 1)],  # task 1 with 3 alternatives
                    [(1, 2), (3, 1)],
                ]
            
            urun55=[  # task = (processing_time, machine_id)
                    [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                    [(3, 0), (3, 1), (3, 2)],  # task 1 with 3 alternatives
                    [(3, 0), (3, 1), (3, 2)],
                ]

            for i in range(order_count):
                if order[i]=='urun1':
                    jobs.append(urun11)
                if order[i]=='urun2':
                    jobs.append(urun22)
                if order[i]=='urun3':
                    jobs.append(urun33)
                if order[i]=='urun4':
                    jobs.append(urun44)
                if order[i]=='urun5':
                    jobs.append(urun55)
            a = 1



            all_jobs = range(order_count)

            num_machines = 4
            all_machines = range(num_machines)

            # Model the flexible jobshop problem.
            model = cp_model.CpModel()


               
            
            jobs =list(jobs)

            for job in range(len(jobs)):
                

                for task in range(len(jobs[job])):
                    
                    
                    for a in range(len(jobs[job][task])):
                       
                        jobs[job][task][a]=list(jobs[job][task][a])
                        
                        jobs[job][task][a][0]=jobs[job][task][a][0]*order2[job]
                        
                        jobs[job][task][a]=tuple(jobs[job][task][a])
                           
            tuple(jobs)

            horizon = 0
            for job in jobs:
                for task in job:
                    max_task_duration = 0
                    for alternative in task:
                        max_task_duration = max(max_task_duration, alternative[0])
                    horizon += max_task_duration            
            
            
            # Global storage of variables.
            intervals_per_machines = collections.defaultdict(list)
            presences_per_machines = collections.defaultdict(list)
            starts_per_machines = collections.defaultdict(list)
            ends_per_machines = collections.defaultdict(list)
            #jobs_per_machines = collections.defaultdict(list)
            jobid_per_machines = collections.defaultdict(list)
            taskid_per_machines2= collections.defaultdict(list)

            starts = {}  # indexed by (job_id, task_id).
            presences = {}  # indexed by (job_id, task_id, alt_id).
            job_ends = []

            # Scan the jobs and create the relevant variables and intervals.
            for job_id in all_jobs:
                job = jobs[job_id]
                num_tasks = len(job)
                previous_end = None
                for task_id in range(num_tasks):
                    task = job[task_id]

                    min_duration = task[0][0]
                    max_duration = task[0][0]

                    num_alternatives = len(task)
                    all_alternatives = range(num_alternatives)

                    for alt_id in range(1, num_alternatives):
                        alt_duration = task[alt_id][0]
                        min_duration = min(min_duration, alt_duration)
                        max_duration = max(max_duration, alt_duration)

                    # Create main interval for the task.
                    suffix_name = '_j%i_t%i' % (job_id, task_id)
                    start = model.NewIntVar(0, horizon, 'start' + suffix_name)
                    duration = model.NewIntVar(min_duration, max_duration,
                                               'duration' + suffix_name)
                    end = model.NewIntVar(0, horizon, 'end' + suffix_name)
                    interval = model.NewIntervalVar(start, duration, end,
                                                    'interval' + suffix_name)

                    # Store the start for the solution.
                    starts[(job_id, task_id)] = start

                    # Add precedence with previous task in the same job.
                    if (previous_end is not None) and (task_id>1) :
                        model.Add(10*starts[(job_id, task_id)] > 13*starts[(job_id, task_id-1)])
                    previous_end = end
                    if task_id==1 and (previous_end is not None) :
                        model.Add(starts[(job_id, task_id)] > 5+starts[(job_id, task_id-1)])
                        previous_end = end
                        

                    # Create alternative intervals.
                    if num_alternatives > 1:
                        l_presences = []
                        for alt_id in all_alternatives:
                            alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                            l_presence = model.NewBoolVar('presence' + alt_suffix)
                            l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                            l_duration = task[alt_id][0]
                            l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                            l_interval = model.NewOptionalIntervalVar(
                                l_start, l_duration, l_end, l_presence,
                                'interval' + alt_suffix)
                            l_presences.append(l_presence)

                            # Link the master variables with the local ones.
                            model.Add(start == l_start).OnlyEnforceIf(l_presence)
                            model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                            model.Add(end == l_end).OnlyEnforceIf(l_presence)

                            # Store the presences for the solution.
                            presences[(job_id, task_id, alt_id)] = l_presence

                            # Add the local interval to the right machine.
                            intervals_per_machines[task[alt_id][1]].append(l_interval)
                            starts_per_machines[task[alt_id][1]].append(l_start)
                            ends_per_machines[task[alt_id][1]].append(l_end)
                            presences_per_machines[task[alt_id][1]].append(l_presence)
                            #jobs_per_machines[task[alt_id][1]].append((job_id, task_id))
                            jobid_per_machines[task[alt_id][1]].append(job_id)
                            taskid_per_machines2[task[alt_id][1]].append((job_id, task_id))

                        # Select exactly one presence variable.
                        model.Add(sum(l_presences) == 1)
                    else:
                        intervals_per_machines[task[0][1]].append(interval)
                        presences[(job_id, task_id, 0)] = model.NewConstant(1)

                job_ends.append(previous_end)

            # Create machines constraints.
            for machine_id in all_machines:
                intervals = intervals_per_machines[machine_id]
                machine_starts = starts_per_machines[machine_id]
                machine_ends = ends_per_machines[machine_id]
                machine_presences = presences_per_machines[machine_id]
                jobidmachines = jobid_per_machines[machine_id]
                taskidmachines2= taskid_per_machines2[machine_id]


                if len(intervals) > 1:
                    model.AddNoOverlap(intervals)


                arcs = []

                # added to avoid errors for arcs without data
                if range(len(intervals)) == range(0, 0):
                    continue
                else:
                    for j1 in range(len(intervals)):

                        # Initial arc from the dummy node (0) to a task.
                        start_lit = model.NewBoolVar('%i is first job' % j1)
                        arcs.append([0, j1 + 1, start_lit])
                        # Final arc from an arc to the dummy node.
                        arcs.append([j1 + 1, 0, model.NewBoolVar('%i is last job' % j1)])

                        # self arc if not present
                        arcs.append([j1+1,j1+1, machine_presences[j1].Not()])

                        for j2 in range(len(intervals)):
                            if j1 == j2:
                                continue

                            lit = model.NewBoolVar('%i follows %i' % (j2, j1))
                            arcs.append([j1 + 1, j2 + 1, lit])

                            # We add the reified precedence to link the literal with the
                            # times of the two tasks.



                            #print(jobidmachines[j1],taskidmachines2[j1][1])



                            if machine_id==0:
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 6).OnlyEnforceIf(lit)


                            if machine_id==1:
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 6).OnlyEnforceIf(lit)

                            if machine_id==2:
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)

                            if machine_id==3:
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                              if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)



                    model.AddCircuit(arcs)

            # Makespan objective
            makespan = model.NewIntVar(0, horizon, 'makespan')
            model.AddMaxEquality(makespan, job_ends)
            model.Minimize(makespan)

            # Solve model.
            run = st.button('Çalıştır')
            if run==True:
                st.subheader('Planalama yapılırken biraz müzik dinlemek ister misiniz ?')

                st.write('erik satie gnossienne no 1')
                audio_file = open('erik.ogg', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                st.write('')

                st.write('chopin nocturne in c sharp minor no. 20')
                audio_file = open('CHOPIN.ogg', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                st.write('')

                st.write('liszt-paganini campanella')
                audio_file = open('Paganini.ogg', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                st.write('')                
                
                solver = cp_model.CpSolver()
                solution_printer = SolutionPrinter()
                solver.parameters.max_time_in_seconds = 60*5
                solver.num_search_workers = 8
                solver.parameters.log_search_progress = True
                status = solver.Solve(model, solution_printer)


                if (status == (4) or status == (2)) and a==1:

                    # Print final solution.
                    df1=pd.DataFrame()
                    for job_id in all_jobs:
                        print('Job %i:' % job_id)
                        for task_id in range(len(jobs[job_id])):
                            start_value = solver.Value(starts[(job_id, task_id)])
                            machine = -1
                            duration = -1
                            selected = -1
                            for alt_id in range(len(jobs[job_id][task_id])):
                                if solver.Value(presences[(job_id, task_id, alt_id)]):
                                    duration = jobs[job_id][task_id][alt_id][0]
                                    machine = jobs[job_id][task_id][alt_id][1]
                                    selected = alt_id
                            print(
                                '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                                (job_id, task_id, start_value, selected, machine, duration))
                            df2= pd.DataFrame([
                              dict(Job="part"+str(job_id)+"-"+"op"+str(task_id),Machine="machine"+str(machine), Start=start_value, Duration=duration,Finish=start_value+duration)])
                
                            df1 = df1.append(df2,ignore_index=True)
                    
                    makine =[0]*num_machines
                    makine[0]='press160'
                    makine[1]='press200'
                    makine[2]='press300'
                    makine[3]='press400'
                    #for i in range(order_count):
                     #   a='job'+str(i)
                      #  df1['Job'] = df1['Job'].str.replace(a,order[i])
                    for i in range(num_machines):
                        b='machine'+str(i)
                        df1['Machine'] = df1['Machine'].str.replace(b,makine[i])                        
                        
                    st.dataframe(df1)
                    print('Solve status: %s' % solver.StatusName(status))
                    print('Optimal objective value: %i' % solver.ObjectiveValue())
                    print()

                    st.write(solver.ObjectiveValue())
                    def visualize(results):

                        schedule = df1
                        JOBS = sorted(list(schedule['Job'].unique()))
                        MACHINES = sorted(list(schedule['Machine'].unique()))
                        makespan = schedule['Finish'].max()

                        bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
                        text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
                        colors = mpl.cm.Dark2.colors

                        schedule.sort_values(by=['Job', 'Start'])
                        schedule.set_index(['Job', 'Machine'], inplace=True)

                        fig, ax = plt.subplots(2,1, figsize=(12, 5+(len(JOBS)+len(MACHINES))/4))

                        for jdx, j in enumerate(JOBS, 1):
                            for mdx, m in enumerate(MACHINES, 1):
                                if (j,m) in schedule.index:
                                    xs = schedule.loc[(j,m), 'Start']
                                    xf = schedule.loc[(j,m), 'Finish']
                                    ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%7], **bar_style)
                                    ax[0].text((xs + xf)/2, jdx, m, **text_style)
                                    ax[1].plot([xs, xf], [mdx]*2, c=colors[jdx%7], **bar_style)
                                    ax[1].text((xs + xf)/2, mdx, j, **text_style)

                        ax[0].set_title('Job-shop Scheduling (JOB PERSPECTIVE)')
                        ax[0].set_ylabel('PARTS')
                        print("")
                        print("")
                        ax[1].set_title('Job-shop Scheduling (MACHINE PERSPECTIVE)')
                        ax[1].set_ylabel('MACHINES')

                        for idx, s in enumerate([JOBS, MACHINES]):
                            ax[idx].set_ylim(0.5, len(s) + 0.5)
                            ax[idx].set_yticks(range(1, 1 + len(s)))
                            ax[idx].set_yticklabels(s)
                            ax[idx].text(makespan, ax[idx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
                            ax[idx].plot([makespan]*2, ax[idx].get_ylim(), 'r--')
                            ax[idx].set_xlabel('Time')
                            ax[idx].grid(True)

                        #fig.tight_layout()

                    
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(visualize(df1))
                    
                    
                    
                    
        
        flexible_jobshop()

    if secenek=='dosya yükleyerek':
        uploaded_file = st.file_uploader("sipariş dosyasını yükleyiniz")
        if uploaded_file is not None:
            df=pd.read_excel(uploaded_file)
            st.dataframe(df)
            a =df.to_numpy()
            for i in range(order_count): 
                 order[i]=a[i][0]
            class SolutionPrinter(cp_model.CpSolverSolutionCallback):


                def __init__(self):
                    cp_model.CpSolverSolutionCallback.__init__(self)
                    self.__solution_count = 0

                def on_solution_callback(self):
                    """Called at each new solution."""
                    print('Solution %i, time = %f s, objective = %i' %
                          (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
                    self.__solution_count += 1


            def flexible_jobshop():

                a = 0
                jobs=[]



                urun11=[  # task = (processing_time, machine_id)
                        [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                        [(3, 0), (5, 1), (2, 2)],  # task 1 with 3 alternatives
                        [(1, 0), (3, 1)],
                    ]



                urun22=[  # task = (processing_time, machine_id)
                        [(3, 0), (3, 1), (4, 2)],  # task 0 with 3 alternatives
                        [(3, 0), (3, 1)],  # task 1 with 2 alternatives
                        [(3, 0), (3, 2)],
                    ]



                urun33=[  # task = (processing_time, machine_id)
                        [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                        [(3, 0), (3, 1), (3, 2)],  # task 1 with 3 alternatives
                        [(3, 0), (3, 1), (3, 2)],
                    ]
                urun44=[  # task = (processing_time, machine_id)
                        [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                        [(3, 0), (3, 1), (3, 2)],  # task 1 with 3 alternatives
                        [(3, 0), (3, 1), (3, 2)],
                    ]                

                urun55=[  # task = (processing_time, machine_id)
                        [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
                        [(3, 0), (3, 1), (3, 2)],  # task 1 with 3 alternatives
                        [(3, 0), (3, 1), (3, 2)],
                    ]


                for i in range(order_count):
                    if order[i]=='urun1':
                        jobs.append(urun11)
                    if order[i]=='urun2':
                        jobs.append(urun22)
                    if order[i]=='urun3':
                        jobs.append(urun33)
                    if order[i]=='urun4':
                        jobs.append(urun44)
                    if order[i]=='urun5':
                        jobs.append(urun55)
                a = 1



                all_jobs = range(order_count)

                num_machines = 3
                all_machines = range(num_machines)

                # Model the flexible jobshop problem.
                model = cp_model.CpModel()

                horizon = 0
                for job in jobs:
                    for task in job:
                        max_task_duration = 0
                        for alternative in task:
                            max_task_duration = max(max_task_duration, alternative[0])
                        horizon += max_task_duration


                # Global storage of variables.
                intervals_per_machines = collections.defaultdict(list)
                presences_per_machines = collections.defaultdict(list)
                starts_per_machines = collections.defaultdict(list)
                ends_per_machines = collections.defaultdict(list)
                #jobs_per_machines = collections.defaultdict(list)
                jobid_per_machines = collections.defaultdict(list)
                taskid_per_machines2= collections.defaultdict(list)

                starts = {}  # indexed by (job_id, task_id).
                presences = {}  # indexed by (job_id, task_id, alt_id).
                job_ends = []

                # Scan the jobs and create the relevant variables and intervals.
                for job_id in all_jobs:
                    job = jobs[job_id]
                    num_tasks = len(job)
                    previous_end = None
                    for task_id in range(num_tasks):
                        task = job[task_id]

                        min_duration = task[0][0]
                        max_duration = task[0][0]

                        num_alternatives = len(task)
                        all_alternatives = range(num_alternatives)

                        for alt_id in range(1, num_alternatives):
                            alt_duration = task[alt_id][0]
                            min_duration = min(min_duration, alt_duration)
                            max_duration = max(max_duration, alt_duration)

                        # Create main interval for the task.
                        suffix_name = '_j%i_t%i' % (job_id, task_id)
                        start = model.NewIntVar(0, horizon, 'start' + suffix_name)
                        duration = model.NewIntVar(min_duration, max_duration,
                                                   'duration' + suffix_name)
                        end = model.NewIntVar(0, horizon, 'end' + suffix_name)
                        interval = model.NewIntervalVar(start, duration, end,
                                                        'interval' + suffix_name)

                        # Store the start for the solution.
                        starts[(job_id, task_id)] = start

                        # Add precedence with previous task in the same job.
                        if previous_end is not None:
                            model.Add(start >= previous_end)
                        previous_end = end

                        # Create alternative intervals.
                        if num_alternatives > 1:
                            l_presences = []
                            for alt_id in all_alternatives:
                                alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                                l_presence = model.NewBoolVar('presence' + alt_suffix)
                                l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                                l_duration = task[alt_id][0]
                                l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                                l_interval = model.NewOptionalIntervalVar(
                                    l_start, l_duration, l_end, l_presence,
                                    'interval' + alt_suffix)
                                l_presences.append(l_presence)

                                # Link the master variables with the local ones.
                                model.Add(start == l_start).OnlyEnforceIf(l_presence)
                                model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                                model.Add(end == l_end).OnlyEnforceIf(l_presence)

                                # Store the presences for the solution.
                                presences[(job_id, task_id, alt_id)] = l_presence

                                # Add the local interval to the right machine.
                                intervals_per_machines[task[alt_id][1]].append(l_interval)
                                starts_per_machines[task[alt_id][1]].append(l_start)
                                ends_per_machines[task[alt_id][1]].append(l_end)
                                presences_per_machines[task[alt_id][1]].append(l_presence)
                                #jobs_per_machines[task[alt_id][1]].append((job_id, task_id))
                                jobid_per_machines[task[alt_id][1]].append(job_id)
                                taskid_per_machines2[task[alt_id][1]].append((job_id, task_id))

                            # Select exactly one presence variable.
                            model.Add(sum(l_presences) == 1)
                        else:
                            intervals_per_machines[task[0][1]].append(interval)
                            presences[(job_id, task_id, 0)] = model.NewConstant(1)

                    job_ends.append(previous_end)

                # Create machines constraints.
                for machine_id in all_machines:
                    intervals = intervals_per_machines[machine_id]
                    machine_starts = starts_per_machines[machine_id]
                    machine_ends = ends_per_machines[machine_id]
                    machine_presences = presences_per_machines[machine_id]
                    jobidmachines = jobid_per_machines[machine_id]
                    taskidmachines2= taskid_per_machines2[machine_id]


                    if len(intervals) > 1:
                        model.AddNoOverlap(intervals)


                    arcs = []

                    # added to avoid errors for arcs without data
                    if range(len(intervals)) == range(0, 0):
                        continue
                    else:
                        for j1 in range(len(intervals)):

                            # Initial arc from the dummy node (0) to a task.
                            start_lit = model.NewBoolVar('%i is first job' % j1)
                            arcs.append([0, j1 + 1, start_lit])
                            # Final arc from an arc to the dummy node.
                            arcs.append([j1 + 1, 0, model.NewBoolVar('%i is last job' % j1)])

                            # self arc if not present
                            arcs.append([j1+1,j1+1, machine_presences[j1].Not()])

                            for j2 in range(len(intervals)):
                                if j1 == j2:
                                    continue

                                lit = model.NewBoolVar('%i follows %i' % (j2, j1))
                                arcs.append([j1 + 1, j2 + 1, lit])

                                # We add the reified precedence to link the literal with the
                                # times of the two tasks.



                                #print(jobidmachines[j1],taskidmachines2[j1][1])



                                if machine_id==0:
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 6).OnlyEnforceIf(lit)


                                if machine_id==1:
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 6).OnlyEnforceIf(lit)

                                if machine_id==2:
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                                  if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                                    model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)





                        model.AddCircuit(arcs)

                # Makespan objective
                makespan = model.NewIntVar(0, horizon, 'makespan')
                model.AddMaxEquality(makespan, job_ends)
                model.Minimize(makespan)

                # Solve model.
                run = st.button('Çalıştır')
                if run==True:
                    st.subheader('Planalama yapılırken biraz müzik dinlemek ister misiniz ?')

                    st.write('erik satie gnossienne no 1')
                    audio_file = open('erik.ogg', 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/ogg')
                    st.write('')

                    st.write('chopin nocturne in c sharp minor no. 20')
                    audio_file = open('CHOPIN.ogg', 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/ogg')
                    st.write('')

                    st.write('liszt-paganini campanella')
                    audio_file = open('Paganini.ogg', 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/ogg')
                    st.write('')                

                    solver = cp_model.CpSolver()
                    solution_printer = SolutionPrinter()
                    solver.parameters.max_time_in_seconds = 60*5
                    solver.num_search_workers = 8
                    solver.parameters.log_search_progress = True
                    status = solver.Solve(model, solution_printer)


                    if (status == (4) or status == (2)) and a==1:
                                        
                        # Print final solution.
                        df1=pd.DataFrame()
                        for job_id in all_jobs:
                            print('Job %i:' % job_id)
                            for task_id in range(len(jobs[job_id])):
                                start_value = solver.Value(starts[(job_id, task_id)])
                                machine = -1
                                duration = -1
                                selected = -1
                                for alt_id in range(len(jobs[job_id][task_id])):
                                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                                        duration = jobs[job_id][task_id][alt_id][0]
                                        machine = jobs[job_id][task_id][alt_id][1]
                                        selected = alt_id
                                print(
                                    '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                                    (job_id, task_id, start_value, selected, machine, duration))
                                df2= pd.DataFrame([dict(Job="part"+str(job_id +1)+"-"+"op"+str(task_id+1),Machine="machine"+str(machine), Start=start_value, Duration = duration,Finish=start_value+duration)])
                
                                df1 = df1.append(df2,ignore_index=True)
                        makine =[0]*num_machines
                        makine[0]='press160'
                        makine[1]='press200'
                        makine[2]='press300'
                        for i in range(order_count):
                            a='job'+str(i)
                            df1['Job'] = df1['Job'].str.replace(a,order[i])
                        for i in range(num_machines):
                            b='machine'+str(i)
                            df1['Machine'] = df1['Machine'].str.replace(b,makine[i])                        

                        st.dataframe(df1)
                    
                        print('Solve status: %s' % solver.StatusName(status))
                        print('Optimal objective value: %i' % solver.ObjectiveValue())
                        print()

                        st.write(solver.ObjectiveValue())
                        def visualize(results):

                            schedule = df1
                            JOBS = sorted(list(schedule['Job'].unique()))
                            MACHINES = sorted(list(schedule['Machine'].unique()))
                            makespan = schedule['Finish'].max()

                            bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
                            text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
                            colors = mpl.cm.Dark2.colors

                            schedule.sort_values(by=['Job', 'Start'])
                            schedule.set_index(['Job', 'Machine'], inplace=True)

                            fig, ax = plt.subplots(2,1, figsize=(12, 5+(len(JOBS)+len(MACHINES))/4))

                            for jdx, j in enumerate(JOBS, 1):
                                for mdx, m in enumerate(MACHINES, 1):
                                    if (j,m) in schedule.index:
                                        xs = schedule.loc[(j,m), 'Start']
                                        xf = schedule.loc[(j,m), 'Finish']
                                        ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%7], **bar_style)
                                        ax[0].text((xs + xf)/2, jdx, m, **text_style)
                                        ax[1].plot([xs, xf], [mdx]*2, c=colors[jdx%7], **bar_style)
                                        ax[1].text((xs + xf)/2, mdx, j, **text_style)

                            ax[0].set_title('Job-shop Scheduling (JOB PERSPECTIVE)')
                            ax[0].set_ylabel('PARTS')
                            print("")
                            print("")
                            ax[1].set_title('Job-shop Scheduling (MACHINE PERSPECTIVE)')
                            ax[1].set_ylabel('MACHINES')

                            for idx, s in enumerate([JOBS, MACHINES]):
                                ax[idx].set_ylim(0.5, len(s) + 0.5)
                                ax[idx].set_yticks(range(1, 1 + len(s)))
                                ax[idx].set_yticklabels(s)
                                ax[idx].text(makespan, ax[idx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
                                ax[idx].plot([makespan]*2, ax[idx].get_ylim(), 'r--')
                                ax[idx].set_xlabel('Time')
                                ax[idx].grid(True)

                            #fig.tight_layout()


                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot(visualize(df1))
                        

            flexible_jobshop()




elif choose== "Source Code":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: teal;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Source Code</p>', unsafe_allow_html=True)
    code='''
    import collections

from ortools.sat.python import cp_model


        class SolutionPrinter(cp_model.CpSolverSolutionCallback):


            def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.__solution_count = 0

        def on_solution_callback(self):
            """Called at each new solution."""
            print('Solution %i, time = %f s, objective = %i' %
              (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
            self.__solution_count += 1


        def flexible_jobshop():

            jobs=[]

    
            job1=[  # task = (processing_time, machine_id)
            [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
            [(3, 0), (5, 1), (2, 2)],  # task 1 with 3 alternatives
            [(1, 0), (3, 1)],
                ]
                    for i in range(0,1):
                jobs.append(job1)
    
    
                job2=[  # task = (processing_time, machine_id)
            [(3, 0), (3, 1), (4, 2)],  # task 0 with 3 alternatives
            [(3, 0), (3, 1)],  # task 1 with 2 alternatives
            [(3, 0), (3, 2)],
                ]
            for i in range(0,1):
                jobs.append(job2)
    
    
            job3=[  # task = (processing_time, machine_id)
            [(3, 0), (3, 1), (3, 2)],  # task 0 with 3 alternatives
            [(3, 0), (3, 1), (3, 2)],  # task 1 with 3 alternatives
            [(3, 0), (3, 1), (3, 2)],
            ]
    
        for i in range(0,1):
        jobs.append(job3)
    


        num_jobs = len(jobs)
        all_jobs = range(num_jobs)

        num_machines = 3
        all_machines = range(num_machines)

        # Model the flexible jobshop problem.
        model = cp_model.CpModel()

        horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration


    # Global storage of variables.
    intervals_per_machines = collections.defaultdict(list)
    presences_per_machines = collections.defaultdict(list)
    starts_per_machines = collections.defaultdict(list)
    ends_per_machines = collections.defaultdict(list)
    #jobs_per_machines = collections.defaultdict(list)
    jobid_per_machines = collections.defaultdict(list)

    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                    # Add the local interval to the right machine.
                    intervals_per_machines[task[alt_id][1]].append(l_interval)
                    starts_per_machines[task[alt_id][1]].append(l_start)
                    ends_per_machines[task[alt_id][1]].append(l_end)
                    presences_per_machines[task[alt_id][1]].append(l_presence)
                    #jobs_per_machines[task[alt_id][1]].append((job_id, task_id))
                    jobid_per_machines[task[alt_id][1]].append(job_id)

                # Select exactly one presence variable.
                model.Add(sum(l_presences) == 1)
            else:
                intervals_per_machines[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

        # Create machines constraints.
        for machine_id in all_machines:
        intervals = intervals_per_machines[machine_id]
        machine_starts = starts_per_machines[machine_id]
        machine_ends = ends_per_machines[machine_id]
        machine_presences = presences_per_machines[machine_id]
        jobidmachines = jobid_per_machines[machine_id]
        

        if len(intervals) > 1:
            model.AddNoOverlap(intervals)


        arcs = []

        # added to avoid errors for arcs without data
        if range(len(intervals)) == range(0, 0):
            continue
        else:
            for j1 in range(len(intervals)):

                # Initial arc from the dummy node (0) to a task.
                start_lit = model.NewBoolVar('%i is first job' % j1)
                arcs.append([0, j1 + 1, start_lit])
                # Final arc from an arc to the dummy node.
                arcs.append([j1 + 1, 0, model.NewBoolVar('%i is last job' % j1)])

                # self arc if not present
                arcs.append([j1+1,j1+1, machine_presences[j1].Not()])

                for j2 in range(len(intervals)):
                    if j1 == j2:
                        continue

                    lit = model.NewBoolVar('%i follows %i' % (j2, j1))
                    arcs.append([j1 + 1, j2 + 1, lit])

                    # We add the reified precedence to link the literal with the
                    # times of the two tasks.



                    if machine_id==0:
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 6).OnlyEnforceIf(lit)
                      
                    
                    if machine_id==1:
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 6).OnlyEnforceIf(lit)
                    
                    if machine_id==2:
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==0):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 1).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==2) or (jobidmachines[j2] ==2 and jobidmachines[j1] ==0):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 2).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==2 and jobidmachines[j1] ==1) or (jobidmachines[j2] ==1 and jobidmachines[j1] ==2):
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==0 and jobidmachines[j1] ==0) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 4).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==1 and jobidmachines[j1] ==1) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 5).OnlyEnforceIf(lit)
                      if (jobidmachines[j2] ==2 and jobidmachines[j1] ==2) :
                        model.Add(machine_starts[j2] >= machine_ends[j1] + 3).OnlyEnforceIf(lit)
                    
                    

                    

            model.AddCircuit(arcs)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve model.
    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter()
    solver.parameters.max_time_in_seconds = 60*5
    solver.num_search_workers = 8
    solver.parameters.log_search_progress = True
    status = solver.Solve(model, solution_printer)


    if status == (4) or status == (2):

        # Print final solution.
        for job_id in all_jobs:
            print('Job %i:' % job_id)
            for task_id in range(len(jobs[job_id])):
                start_value = solver.Value(starts[(job_id, task_id)])
                machine = -1
                duration = -1
                selected = -1
                for alt_id in range(len(jobs[job_id][task_id])):
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        duration = jobs[job_id][task_id][alt_id][0]
                        machine = jobs[job_id][task_id][alt_id][1]
                        selected = alt_id
                print(
                    '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                    (job_id, task_id, start_value, selected, machine, duration))
                    

        print('Solve status: %s' % solver.StatusName(status))
        print('Optimal objective value: %i' % solver.ObjectiveValue())
        print()




        flexible_jobshop()'''
    st.code(code, language="python")


elif choose == "Extras":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: teal;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Extras</p>', unsafe_allow_html=True)
    st.write('Complete Github repo for this project')
    st.caption('https://github.com/lordensar/kumtel', unsafe_allow_html=False)
    st.write('')
    st.write('Google OR-Tools Documentation')
    st.caption('https://developers.google.com/optimization', unsafe_allow_html=False)
    st.write('')
    st.write('How to deploy a web application in python?')
    st.caption('https://www.sitepoint.com/python-web-applications-the-basics-of-wsgi/', unsafe_allow_html=False)
    st.write('')
    st.write('How to rent virtual computers to run own computer applications with AWS?')
    st.caption('https://aws.amazon.com/tr/ec2/', unsafe_allow_html=False)
    st.write('')
    a = st.selectbox('Do you wanna see most handsome man in the world?',options=['yess','NO???'],index=1)
    if a =='yess':
        akgun = Image.open('akgun.jpeg')
        st.image(akgun)
        

    
    
elif choose == "Contact":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: teal;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        
        Name=st.text_input(label='Please Enter Your Name') 
        Email=st.text_input(label='Please Enter Your Email') 
        Message=st.text_input(label='Please Enter Your Message') 
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Eğer istersek size en yakın zamanda dönüş yapacağız :D')
    
    