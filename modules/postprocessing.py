'''Postprocessing module to implement graph plot and file output'''

# Importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, mkdir, remove, system
from md2pdf.core import md2pdf


class Report():
    '''Markdown Wrapper Class to implement matplolib graphs and markdown syntax easily'''

    def __init__(self,metadata:dict = None, generate_pdf:bool = False):
        self.metadata = metadata
        self.generate_pdf = generate_pdf

    def __enter__(self, filepath:str = "./report.md"):
        self.filepath = filepath
        self.report_file = open(filepath, 'w',  encoding="utf-8")
        self.img_directory = path.dirname(filepath) + "/img/"

        #If directory not exists, will be created
        try:
            mkdir(self.img_directory)
        except FileExistsError:
            print("Directory already exists")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.report_file.close()
        #Generate pdf report if generate_pdf is selected
        if self.generate_pdf:
            directory = path.dirname(path.abspath(self.filepath))
            filename, _ = path.splitext(path.basename(self.filepath))
            md2pdf(f"{directory}/{filename}.pdf",
            md_content=None,
            md_file_path=self.filepath,
            css_file_path="./templates/skeleton.css",
            base_url=directory)


    @staticmethod
    def parse_image(img_path:str, title:str = ""):
        return f'![{title}]({img_path})'

    @staticmethod
    def parse_title(title:str, h:int = 1):
        h_title = "#" * h
        return h_title + " " + title
    
    @staticmethod
    def parse_list(listmd:list, unordered:str = ""):
        output = ""
        # If is an unordered list, argument must include one of the following characters: -,*,+
        if unordered:
            for element in listmd:
                output += unordered + " " + str(element) + "\n"
        else:
            for element in listmd:
                output += str(listmd.index(element)+1) + ". " + str(element) + "\n"
        return output
    
    @staticmethod
    def parse_code(codeblock:str, language:str = "python"):
        return f"```{language}\n{codeblock}\n```"
    

    #Print methods to write markdown report 
    def print(self, paragraph:str):
        self.report_file.write(f"{paragraph}n")

    def print_line(self):
        self.report_file.write("\n---\n")

    def print_title(self,title:str, h:int = 1):
        self.report_file.write(f"{self.parse_title(title,h)}\n")
    
    def print_code(codeblock:str, language:str = "python"):
        self.report_file.write(f"{self.parse_code(codeblock, language)}\n")

    def print_boxplot(self, data:pd.DataFrame, labels:list, filename:str = "boxplot.png", img_title:str = "Box Plot", figsize:tuple = (15,7), color:str = "green", orient:str = "v", **kwargs):
        '''Write to file the boxplot of the given data'''
        fig = plt.figure(figsize=figsize)
        for label in labels:
            ax = fig.add_subplot(1,len(labels), labels.index(label)+1)
            sns.boxplot(y=data[label], color=color, orient=orient)
            fig.tight_layout()
        
        # Saving image to file
        img_path = self.img_directory + filename
        fig.savefig(img_path, **kwargs)

        #Include image in the report
        self.report_file.write(self.parse_image(img_path, img_title))

    

