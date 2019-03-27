import numpy as np
import sys

def checkForFreeVariables (tableaux, variablesType, columnToAnalize):
    if variablesType[columnToAnalize] == 0:
        newColumn = tableaux[:,columnToAnalize]
        newColumn = list(map(lambda x: x * (-1), newColumn))
        newColumn = np.vstack(newColumn)
        tableaux = np.hstack((tableaux, newColumn))
    return tableaux

def addSlackVariable (tableaux, sign, lineToAddSlackVariable, columns):
    if "<" in sign:
        newColumn = np.zeros(len(tableaux))
        newColumn[lineToAddSlackVariable] = 1.0
        newColumn = np.vstack(newColumn)
        tableaux = np.hstack((tableaux, newColumn))       
    elif ">" in sign:
        newColumn = np.zeros(len(tableaux))
        newColumn[lineToAddSlackVariable] = -1.0
        newColumn = np.vstack(newColumn)
        tableaux = np.hstack((tableaux, newColumn))
    return tableaux

def addRestriction (f, tableaux, columns, lineToEdit):
    restriction = f.readline().split()
    for i in range (0, columns):
        tableaux[lineToEdit][i] = restriction[i]
    tableaux[lineToEdit][columns] = restriction[columns+1]
    tableaux = addSlackVariable(tableaux, restriction[columns], lineToEdit, columns)
    return tableaux
    
def fillTableauxObjectiveFunction (objectiveFunction, tableaux):
    for i in range (0, len(objectiveFunction)):
        tableaux[0][i] = objectiveFunction[i]
    tableaux[0,:] = list(map(lambda x: x * (-1), tableaux[0,:]))
    return tableaux

def createMatrix (lines, columns):
    return np.zeros((lines, columns))

def readAndSetAugmentedForm():
    with open(sys.argv[1]) as f:
        variablesCounter = int(f.readline())
        restrictionsCounter = int(f.readline())
        # Tell which is the index of column b
        bIndex = variablesCounter
        tableaux = createMatrix(restrictionsCounter+1, variablesCounter+1)
        variablesType = list(map(float, (f.readline()).split()))
        objectiveFunction = list(map(float, (f.readline()).split()))
        tableaux = fillTableauxObjectiveFunction(objectiveFunction, tableaux)
        for i in range (0, restrictionsCounter):
            tableaux = addRestriction(f, tableaux, variablesCounter, i+1)
        for i in range (0, variablesCounter):
            tableaux = checkForFreeVariables(tableaux, variablesType, i)
    return tableaux, bIndex

def auxProblem (auxTableaux, bIndex):
    auxTableaux[0,:] = list(map(lambda x: x - x, auxTableaux[0,:]))
    for i in range (0, len(auxTableaux)):
        if auxTableaux[i][bIndex] < 0:
            auxTableaux[i,:] = list(map(lambda x: x * (-1), auxTableaux[i,:]))
        else:
            continue
    identityObjectiveFunction = np.zeros(len(auxTableaux) - 1)
    identityObjectiveFunction = list(map(lambda x: x + 1, identityObjectiveFunction))
    identity = np.identity(len(auxTableaux) - 1)
    identity = np.vstack((identityObjectiveFunction, identity))
    auxTableaux = np.hstack((auxTableaux, identity))
    for i in range (1, len(auxTableaux)):
        auxTableaux[0,:] = list(map(lambda x, y: x - y, auxTableaux[0,:], auxTableaux[i,:]))
    auxTableaux, base = applyAuxProblem(auxTableaux, bIndex)
    return auxTableaux, base
        
def applyAuxProblem (auxTableaux, bIndex):
    auxTableaux, base = symplex(auxTableaux, bIndex)
    if auxTableaux[0][bIndex] != 0:
        with open(sys.argv[2], 'w') as o:
            o.write('Status: inviavel\n')
            o.write('Certificado:\n')
            o.write('\n')
            exit(0)
    return auxTableaux, base

def symplex (tableaux, bIndex):
    base = np.zeros(len(tableaux[0]))
    base = list(map(lambda x: x - 1, base))
    flagStop = False
    
    while(flagStop == False):
        counter = 0
        for i in list(range(0, len(tableaux[0]))):
            if i == bIndex or tableaux[0][i] >= 0:
                counter = counter + 1
                continue
            else:
                foundPivot, pivotLine = choosePivot(tableaux, bIndex, i)
                if foundPivot == False:
                    with open(sys.argv[2], 'w') as o:
                        o.write('Status: ilimitado\n')
                        o.write('Certificado:\n')
                        o.write('\n')
                        exit(0)
                else:
                    addToBase(base, pivotLine, i)
                    tableaux = pivotWholeColumn(tableaux, pivotLine, bIndex, i)
        if(counter == len(tableaux[0])):
            flagStop = True
        
    return tableaux, base


def addToBase (base, pivotLine, pivotColumn):
        base[pivotColumn] = pivotLine

def choosePivot (tableaux, bIndex, pivotColumn):
    minimum = 0
    lineToPivot = 0
    flagFirst = True
    for i in range (1, len(tableaux)):
        if flagFirst == True and tableaux[i][pivotColumn] > 0:
            minimum = tableaux[i][bIndex] / tableaux[i][pivotColumn]
            minimum = tooSmall(minimum)
            lineToPivot = i
            flagFirst = False
        elif flagFirst == False and tableaux[i][pivotColumn] > 0:
            if tableaux[i][bIndex] / tableaux[i][pivotColumn] < minimum:
                minimum = tableaux[i][bIndex] / tableaux[i][pivotColumn]
                minimum = tooSmall(minimum)
                lineToPivot = i
    if flagFirst == True:
        return False, 0
    else:
        return True, lineToPivot

def tooSmall (number):
    if abs(number) < 0.01:
        number = 0
    return number

def pivotWholeColumn (tableaux, pivotLine, bIndex, pivotColumn):
    divisor = tableaux[pivotLine][pivotColumn]
    tableaux[pivotLine, :] = list(map(lambda x: tooSmall(x / divisor), tableaux[pivotLine, :]))
    for i in range (0, len(tableaux)):
        if i != pivotLine:
            tableaux[i] = tableaux[i] - (tableaux[i][pivotColumn] * tableaux[pivotLine])

    return tableaux
    
def mergeAuxiliarIntoOriginal (tableaux, auxTableaux, base):
    identityStartColumn = ((len(auxTableaux[0])) - (len(auxTableaux) - 1))
    auxTableaux = auxTableaux[1:,:identityStartColumn]
    tableaux = tableaux[:1]
    tableaux = np.vstack((tableaux, auxTableaux))
    tableaux = canonicalForm(tableaux, base)
    return tableaux

def canonicalForm (tableaux, base):
    for i in range (0, len(tableaux)):
        if base[i] > 0:
            tableaux[0,:] = list(map(lambda x, y: x + y, tableaux[(base[i]),:] * (-1) * tableaux[0][i], tableaux[0,:]))
    return tableaux

def printSolution(tableaux, base, o):
    count = 0
    printable = []
    for i in range (0, len(base)):
        if base[i] > 0:
            count = count + 1
    toPrint = 1
    while(count > 0):
        for j in range (0, len(base)):
            if base[j] == toPrint:
                printable.append(tableaux[toPrint][j])
                count = count - 1
    o.write(str(printable))

def main ():
    np.set_printoptions(precision=2)
    tableaux, bIndex = readAndSetAugmentedForm()
    auxTableaux = np.copy(tableaux)
    auxTableaux, base = auxProblem(auxTableaux, bIndex)
    tableaux = mergeAuxiliarIntoOriginal(tableaux, auxTableaux, base)
    tableaux, invalidBase = symplex(tableaux, bIndex)
    with open(sys.argv[2], 'w') as o:
        o.write('Status: otimo\n')
        o.write('Objetivo: ')
        objetivo = tableaux[0][len(tableaux)]
        o.write(str(objetivo))
        # o.write(tableaux[0][len(tableaux)])
        o.write('\n')
        o.write('Solucao:\n')
        printSolution(tableaux, base, o)
        o.write('\n')
        o.write('Certificado:\n')
        o.write('\n')
    
if __name__ == "__main__":
    main()