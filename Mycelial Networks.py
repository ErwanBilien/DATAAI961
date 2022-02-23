#######################
# AUTEUR : Erwan BILIEN
#######################

import json  # Pour la sérialisation/désérialisation des objects
import math
import random
import string
from collections import defaultdict
from typing import List

import mesa
import mesa.space
import numpy as np
from mesa import Agent, Model

from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization import ModularVisualization
from mesa.visualization.ModularVisualization import VisualizationElement, ModularServer
from mesa.visualization.modules import ChartModule

import uuid  # Génération de Unique ID

####
PROBA_MORT = 0.06 #probabilité de déces d'une spore à chaque étape
PROBA_NOUVELLE_SOURCE=0.1#probabilité d'appartion d'une nouvelle source nourriture à chaque étape
RESERVE_INIT=50 #reserbe initiale des sources de nourriture
COLLABORATION=False #les differentes especes de spores colaborent
####

DISTANCE=20
LARGEUR_MONDE=600
HAUTEUR_MONDE=600



COLLABORATION=True #les differentes especes de spores colaborent

#creer un melange de [-1,0,1]
def iter_array():
    it=[-1,0,1]
    np.random.shuffle(it)
    return(it)

class Monde(mesa.Model):
    def  __init__(self,n_sources,n_spores):
        mesa.Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(LARGEUR_MONDE, HAUTEUR_MONDE, False)
        self.schedule = RandomActivation(self)
        self.nourriture=[]


        for it in range(n_sources):
            x=random.randint(self.space.x_min+DISTANCE,self.space.x_max)
            y=random.randint(self.space.y_min+DISTANCE,self.space.y_max)
            x-=x%DISTANCE
            y-=y%DISTANCE
            for i in [0,1]:
                for j in [0,1]:
                    self.nourriture.append(SourceNourriture(x+i*DISTANCE,y+DISTANCE*j,RESERVE_INIT))
            if it<n_spores:
                self.schedule.add(Spore( x, y, uuid.uuid1(),self,it))
        if n_spores>n_sources:
            for i in range(n_spores-n_sources):
                x=random.randint(self.space.x_min+DISTANCE,self.space.x_max)
                y=random.randint(self.space.y_min+DISTANCE,self.space.y_max)
                x-=x%DISTANCE
                y-=y%DISTANCE
                self.schedule.add(Spore( x, y, uuid.uuid1(),self,n_sources+i))

        self.datacollector = DataCollector(
            model_reporters={"n_spores": lambda model: len(model.schedule.agents),
            "n_sources": lambda model: len(model.nourriture)}
        )


    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

        #mort des spores
        for agent in self.schedule.agents:
            if random.random()<PROBA_MORT:
                self.schedule.remove(agent)

        #suppression des sources de nourriture vides
        stop=False
        i=0
        while not stop:
            if self.nourriture[i].reserve==0:
                self.nourriture.pop(i)
                i=0
            else :
                i+=1
            if i==len(self.nourriture):
                stop=True

        #apparition d'une nouvelle source
        if random.random()<PROBA_NOUVELLE_SOURCE:
            x=random.randint(self.space.x_min+DISTANCE,self.space.x_max)
            y=random.randint(self.space.y_min+DISTANCE,self.space.y_max)
            x-=x%DISTANCE
            y-=y%DISTANCE
            for i in [0,1]:
                for j in [0,1]:
                    self.nourriture.append(SourceNourriture(x+i*DISTANCE,y+DISTANCE*j,RESERVE_INIT))
                    print("New food source at ",x,y)

        if self.schedule.steps >= 200:
            self.running = False

class SourceNourriture:
    def __init__(self, x, y,reserve):
        self.x = x
        self.y = y
        self.reserve=reserve

    def portrayal_method(self):
        color="red"
        if self.reserve<RESERVE_INIT*0.5:
            color="orange"
        elif self.reserve<RESERVE_INIT*0.1:
            color="yellow"
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": 0.6*DISTANCE}
        return portrayal

class Nourriture:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Layer": 3,
                     "Filled": "true",
                     "Color": "yellow",
                     "r": 1}
        return portrayal


class Spore(mesa.Agent):
    def __init__(self, x, y, unique_id: int, model: Monde,espece):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.model = model
        self.indice_transmission=0 #plus self.indice_transmission est petit, plus le spore est proche d'une case vide
        self.stock_nourriture=[] #nourriture possedée par le spore (vide si aucune nourriture)
        self.affichage_nourriture=[] #utilisé pour l'affichage de la nourriture en transit
        self.espece=espece #espece du spore (int)

    def portrayal_method(self):
        couleur_spore=["#9ACD32","#6B8E23","#808000","#556B2F","#ADFF2F","#7FFF00","#7CFC00","#00FF00","#32CD32","#F5FFFA","#F0FFF0"]
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": couleur_spore[self.espece%11],
                     "r": 0.5*DISTANCE}
        return portrayal

    #renvoie les coordonnées d'une case vide autours de (x,y), None si toutes les cases adjacentes sont occupées
    def espace_libre(self):
        x=self.pos[0]
        y=self.pos[1]
        voisinage=np.array([[True,True,True],[True,False,True],[True,True,True]])
        for agent in self.model.schedule.agents:
            for i in iter_array():
                for j in iter_array():
                    #print(agent.pos)
                    if np.abs(agent.pos[0]-x-i*DISTANCE)<0.001 and np.abs(agent.pos[1]-y-j*DISTANCE)<0.001 or x+i*DISTANCE not in np.arange(self.model.space.x_min,self.model.space.x_max) or y+j*DISTANCE not in np.arange(self.model.space.y_min,self.model.space.y_max) :
                        voisinage[1+i,1+j]=False
        for i in iter_array():
            for j in iter_array():
                if voisinage[1+i,1+j]==True :
                    return (x+i*DISTANCE,y+j*DISTANCE)
        return None

    #renvoie les coordonnées du spore adjacent le plus proche d'une case vide
    def transmission(self):
        x=self.pos[0]
        y=self.pos[1]
        valeur=math.inf
        plus_proche=None
        voisinage=np.array([[True,True,True],[True,False,True],[True,True,True]])
        for agent in self.model.schedule.agents:
            for i in iter_array():
                for j in iter_array():
                    if agent.pos==(x+i*DISTANCE,y+j*DISTANCE) and agent.indice_transmission<=valeur and (COLLABORATION==True or self.espece==agent.espece):
                        plus_proche=agent
                        valeur=agent.indice_transmission
        return plus_proche

    def maj_indice_transmission(self):
        case_libre=self.espace_libre()
        self.indice_transmission=math.inf
        if case_libre!=None:
            self.indice_transmission=0;
        else:
            for agent in self.model.schedule.agents:
                for i in iter_array():
                    for j in iter_array():
                        #print(agent.pos)
                        if np.abs(agent.pos[0]-self.pos[0]-i*DISTANCE)<0.001 and np.abs(agent.pos[1]-self.pos[1]-j*DISTANCE)<0.001 and (COLLABORATION==True or self.espece==agent.espece):
                            self.indice_transmission=min(agent.indice_transmission+1,self.indice_transmission)
                            #print("indice transmission ",agent.indice_transmission+1)

    def step(self):
        self.maj_indice_transmission()
        #collecte de nourriture
        for source in self.model.nourriture:
            if np.abs(source.x-self.pos[0])<0.001 and np.abs(source.y-self.pos[1])<0.001:
                self.stock_nourriture.append(Nourriture(source.x,source.y))
                source.reserve+=-1
                print("collect food at ",source.x,source.y)


        self.affichage_nourriture=self.stock_nourriture

        while len(self.stock_nourriture)>0:
            case_libre=self.espace_libre()
            if random.random()<0.001:
                self.stock_nourriture.pop(0)
            #extension du réseau
            elif case_libre!=None:
                self.stock_nourriture.pop(0)
                self.model.schedule.add(Spore(case_libre[0],case_libre[1],uuid.uuid1(),self.model,self.espece))
                print(self.pos)
                print("spore created at ",case_libre[0],case_libre[1])
            #transmission de la nourriture à un autre sport plus proche d'une case vide
            else:
                cible=self.transmission()
                #print("transmission ",cible)
                cible.stock_nourriture.append(self.stock_nourriture.pop(0))
                print("food delivered at ",cible.pos)


class ContinuousCanvas(VisualizationElement):
    local_includes = [
        "./js/simple_continuous_canvas.js",
    ]

    def __init__(self, canvas_height=LARGEUR_MONDE,
                 canvas_width=HAUTEUR_MONDE, instantiate=True):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.identifier = "monde-canvas"
        if (instantiate):
            new_element = ("new Simple_Continuous_Module({}, {},'{}')".
                           format(self.canvas_width, self.canvas_height, self.identifier))
            self.js_code = "elements.push(" + new_element + ");"

    def portrayal_method(self, obj):
        return obj.portrayal_method()

    def render(self, model):
        representation = defaultdict(list)
        for obj in model.schedule.agents:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.pos[0] - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.pos[1] - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
                #print(portrayal["x"],portrayal["y"])
            representation[portrayal["Layer"]].append(portrayal)


            for nourriture in obj.affichage_nourriture:
                portrayal = self.portrayal_method(obj)

                if portrayal:
                    portrayal["x"] = ((nourriture.x - model.space.x_min) /
                                      (model.space.x_max - model.space.x_min))
                    portrayal["y"] = ((nourriture.y - model.space.y_min) /
                                      (model.space.y_max - model.space.y_min))
                    #print(portrayal["x"],portrayal["y"])
                representation[portrayal["Layer"]].append(portrayal)

        for obj in model.nourriture:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        #print(representation)
        return representation


def run_single_server():
    chart = ChartModule([{"Label": "n_spores",
                          "Color": "green"},{"Label": "n_sources",
                          "Color": "red"},
                         ],
                        data_collector_name='datacollector')

    server = ModularServer(Monde,
                           [ContinuousCanvas(), chart],
                           "Monde",
                            {"n_sources": ModularVisualization.UserSettableParameter('slider',
                                                                                    "Nombre de sources de nourriture",
                                                                                    1, 1, 10, 1),
                            "n_spores": ModularVisualization.UserSettableParameter('slider',
                                                                                  "Nombre de spores",
                                                                                  1, 1, 10, 1)}
                           )
    server.port = 8522
    server.launch()


if __name__ == "__main__":
    run_single_server()
