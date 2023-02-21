
import random as rd

import matplotlib.pyplot as plt
import numpy as np




class SudokuObject():
    
    def __init__(self, sudoku_data: list[int]):
        """
        Create an SudokuObject instance. sudoku_data is an 81 element int list. 1-9 are fixed literals, 0 are unknown cells.
        """
        # ._grid is 2D list that stores sudoku data
        self._grid = [list() for _ in range(9)]
        for i, n in enumerate(sudoku_data):
            self._grid[i // 9].append(n)
            
            
    def whatis_rowlist(self, i, j) -> list:
        """
        Return the row of the given coordinate
        """
        return self._grid[i].copy()
    
    
    def whatis_collist(self, i, j) -> list:
        """
        Return the column of the given coordinate
        """       
        return [row[j] for row in self._grid]
    
    
    def whatis_blklist(self, i, j) -> list:
        """
        Return the block of the given coordinate
        """       
        block = []
        
        blk_startcoords = ((i // 3) * 3, (j // 3) * 3)
        for blk_i in range(blk_startcoords[0], blk_startcoords[0] + 3):
            for blk_j in range(blk_startcoords[1], blk_startcoords[1] + 3):
                block.append(self._grid[blk_i][blk_j])
                
        return block
    
    
    def update(self, i, j, n):
        """
        Change a cell on sudoku
        """
        self._grid[i][j] = n 
    
    
    def display(self):
        """
        Display the sudoku
        """
        display_str = ""
        
        for i in range(9):
            
            if i > 0 and i % 3 == 0:
                display_str += "\n" + "--" * 11 + "\n"
            else:
                display_str += "\n"
                      
            for j in range(9):
            
                if j > 0 and j % 3 == 0:
                    display_str += " |"
                display_str += " " + str(self._grid[i][j])
            
        print(display_str)
        
    
    def related_coords(self, i, j) -> set[tuple[int, int]]:
        """
        Return a set of coordinates that are related to (in the same row, column or block) given coordinate.
        """
        
        # Constructing the row and column
        rel_row = [(i, rel_j) for rel_j in range(9) if rel_j != j]
        rel_col = [(rel_i, j) for rel_i in range(9) if rel_i != i]
        
        # And constructing the block
        rel_blk = []
        blk_startcoords = ((i // 3) * 3, (j // 3) * 3)
        for blk_i in range(blk_startcoords[0], blk_startcoords[0] + 3):
            for blk_j in range(blk_startcoords[1], blk_startcoords[1] + 3):
                if blk_i != i and blk_j != j:
                    rel_blk.append((blk_i, blk_j))     
        
        # Creating a new set object. A set is an unordered collection of uniqe elements
        related = set(rel_row + rel_col + rel_blk)
        return related
    
    
    def return_grid(self) -> list[list[int]]:
        """
        Return the sudoku grid using a 2D list
        """
        return self._grid
    

    def copy(self):
        """
        Return a copy of SudokuObject instance
        """
        sudoku_list = []
        for i in range(9):
            for j in range(9):
                sudoku_list.append(self._grid[i][j])
        return SudokuObject(sudoku_list)

        

class SudokuPencilMark():
    
    def __init__(self, sudoku: SudokuObject):
        """
        Create an helper object for doing pencil mark operation on the given SudokuObject instance
        """
        self._sudoku = sudoku
        
        # In it's initial state, each cell has 9 potential values. These possibilities are represented
        # using 9 boolean values for each cell. And they are ordered in a grid, same way as sudoku.
        # Initial boolean values for cells are all True's
        self._possible_values_grid = [[[True for _ in range(9)] for _ in range(9)] for _ in range(9)]
        
        
    def run(self):
        """
        Run pencil mark operation (algorithm) on the given sudoku, by modifying cell values
        """
        while True:
            
            # This is a counter. It is used to determine when to stop the pencil mark operation.
            # It counts hidden values found each loop. If counter stays zero at the end, that means
            # the operation was in vain, unable to find any values. When it happens, loop breaks.
            values_found = 0
            
            # Checking fixed values on the sudoku table, and changing boolean data according to them.
            # Changing boolean data of both fixed and related coordinates.
            for i in range(9):
                for j in range(9):
                    
                    val = self._sudoku.return_grid()[i][j]
                    
                    # If cell has a value, then turn all its boolean to False except its current value.
                    # Then, iterate over its related coordinates, and turn the bool that represents cell value to False.
                    if val != 0:
                        self._possible_values_grid[i][j] = [False for _ in range(9)]
                        self._possible_values_grid[i][j][val-1] = True
                        for coord in self._sudoku.related_coords(i, j):
                            self._possible_values_grid[coord[0]][coord[1]][val-1] = False
            
            # Updating the sudoku table by boolean data.
            for i in range(9):
                for j in range(9):
                    
                    val = self._sudoku.return_grid()[i][j]
                    cell = self._possible_values_grid[i][j]
                    
                    # If a previously unknown cell now only has one potential value (only has one True), then 
                    # update the sudoku with this new value and increase the counter. 
                    if cell.count(True) == 1 and val == 0:
                        self._sudoku.update(i, j, cell.index(True)+1)
                        values_found += 1
                    
            if values_found == 0:
                break    



class SudokuGA():
    
    def __init__(self, initialsudoku: SudokuObject):
        """
        Create an helper object for solving sudokus with genetic algorithms
        """
        
        # Genetic algorithm variables
        self._best_select_perc = 0.10       # Top A % of sorted population will generate children
        self._rand_select_perc = 0.10       # Random B % of population will also generate children, to increase diversity 
        self._num_of_children = 10          # Number of children after children generation
        self._population_size = 1000        # Size of population
        self._mutation_perc = 0.60          # X % of the population will have mutations
        self._mutation_max_num_cells = 5    # Maximum N number of their cells will mutate
        self._max_cycles = 100              # Algorithm will restart if unable to find any solution after M cycles
        self._max_restart = 5               # Program will crash, if algorithm cannot find a solution in N restarts
        
        # If equation is wrong, that means the population will not stay the same. It will either increase or decrease.
        if ((self._best_select_perc + self._rand_select_perc) / 2) * self._num_of_children != 1:
            raise Exception("Population is not stable\n((best_select + rand_select) / 2) * num_child\nshould be equal 1")
        
        # Sum of best and random selection percentages should be an even number
        if self._population_size % 2 != 0:
            raise Exception("Sum of best selection percentage and random selection percentages should be an even integer. Otherwise, population wouldn't be stable.")
        
        # Coordinates of fixed (already known) values. 
        self._fixed = []
        for i in range(9):
            for j in range(9):
                
                if initialsudoku.return_grid()[i][j] != 0:
                    self._fixed.append((i, j))
                    
        # Initial sudoku
        self._init_sudoku = initialsudoku
        
    
    def run(self, info: bool = False):
        """
        Start genetic algorithm. Solves the sudoku if possible and returns the solution. Otherwise returns an exception.
        When info set True, provides additional information. 
        """
        
        # Used for plotting
        if info:
            best_elements = []
            worst_elements = []
            cycle_counter = 0
            
        for _ in range(self._max_restart):
            
            # Creating the first random population
            population = []
            for _ in range(self._population_size):
                new_sudoku = self._init_sudoku.copy()
                self._rand_fill_sudoku(new_sudoku)
                population.append(new_sudoku)
            
            for _ in range(self._max_cycles):
                
                # Sorting the population by their fitness values, from low to high (lower fitness values are better)
                population.sort(key=self._fitness_filled_sudoku)
                
                # Inform the user
                if info:
                    cycle_counter += 1
                    print(f"Cycle: {cycle_counter}, Best: {self._fitness_filled_sudoku(population[0])}, Worst: {self._fitness_filled_sudoku(population[-1])}")
                    best_elements.append(self._fitness_filled_sudoku(population[0]))
                    worst_elements.append(self._fitness_filled_sudoku(population[-1]))
                
                # Zero fitness means solution, return the solution if that exists
                if self._fitness_filled_sudoku(population[0]) == 0:
                    
                    # Plotting data
                    if info:
                        fig, ax = plt.subplots()
                        ax.set_title("Genetic Algorithm Fitness Values")
                        ax.set_ylabel("Fitness")
                        ax.set_xlabel("Cycles")
                        ax.plot(best_elements, label="best case")
                        ax.plot(worst_elements, label= "worst case")
                        ax.legend()
                        plt.show()
                    
                    return population[0]
                    
                # Selecting the parents. First selecting best ones, then selecting randomly
                num_of_best = int(self._best_select_perc * self._population_size)   # Top K elements of population
                num_of_rand = int(self._rand_select_perc * self._population_size)   # And L other random element
                parents_list = population[:num_of_best]                             # Taking top K elements as parents,
                parents_list += rd.sample(population[num_of_best:], num_of_rand)    # And L random unique element that are not in top K
                
                # Randomly select two parents, generate children
                rd.shuffle(parents_list)                                            
                children_list = []
                
                for i in range(0, len(parents_list), 2):
                    children_list += self._generate_children(parents_list[i], parents_list[i+1])
                    
                # Mutate some children to increase the diversity even further
                num_mutation = int(self._mutation_perc * self._population_size)
                for child in rd.sample(children_list, num_mutation):
                    self._mutate(child)

                # Turn children_list to new population and repeat
                population = children_list
                
        # Plotting data
        if info:
            fig, ax = plt.subplots()
            ax.set_title("Genetic Algorithm Fitness Values")
            ax.set_ylabel("Fitness")
            ax.set_xlabel("Cycles")
            ax.plot(best_elements, label="best case")
            ax.plot(worst_elements, label= "worst case")
            ax.legend()
            plt.show() 
                 
        raise Exception("Couldn't find a solution") 
        
    
    def _rand_fill_sudoku(self, initsudoku: SudokuObject):
        """
        Fill empty cells of a given sudoku with random numbers
        """
        for i in range(9):
            
            pos_values = [n for n in range(1, 10) if n not in initsudoku.whatis_rowlist(i, 0)]
            
            for j in range(9):
                
                if initsudoku.return_grid()[i][j] == 0:
                    
                    n = rd.choice(pos_values)
                    initsudoku.update(i, j, n)
                    pos_values.remove(n)
    
    
    def _fitness_filled_sudoku(self, sudoku: SudokuObject) -> int:
        """
        Measures how far the sudoku from its solution. Works only for sudokus that have
        no empty cells (that are filled), raises an exception otherwise. The measurement is 
        called "the fitness of sudoku". Lower this number, closer to a solution.
        """
            
        fitness = 0

        # Processing rows' fitness 
        for i in range(9):
            
            row = sudoku.whatis_rowlist(i, 0)
            for n in range(1, 10):
                if n not in row: fitness += 1

        # Processing columns' fitness
        for j in range(9):
            
            col = sudoku.whatis_collist(0, j)
            for n in range(1, 10):
                if n not in col: fitness += 1        
        
        # Processing blocks' fitness
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                
                blk = sudoku.whatis_blklist(i, j)
                for n in range(1, 10):
                    if n not in blk: fitness += 1
                    
        return fitness
    
    
    def _mutate(self, sudoku: SudokuObject):
        """
        Mutate the sudoku. Zero to determined number of non-fixed values change.
        """
        for _ in range(self._mutation_max_num_cells):
            
            i = rd.randint(0, 8)
            j1 = rd.randint(0, 8)
            j2 = rd.randint(0, 8)
            
            # Randomly select two non-fixed values on a row and swap them
            if (i, j1) not in self._fixed and (i, j2) not in self._fixed:
                
                n = sudoku.return_grid()[i][j1]
                m = sudoku.return_grid()[i][j2]
                sudoku.update(i, j1, m)
                sudoku.update(i, j2, n)
            
            
    def _generate_children(self, sudoku1: SudokuObject, sudoku2: SudokuObject) -> list[SudokuObject]:
        """
        Using two sudokus, create determined number of new sudokus by randomly taking 
        (50/50 for each row, like flipping a coin) rows from parents.
        """
        
        children_sudoku_list = []
        
        for _ in range(self._num_of_children):
            
            child_list = []
            for i in range(9):
                
                n = rd.randint(1, 2)
                if n == 1:
                    child_list += sudoku1.whatis_rowlist(i, 0)
                if n == 2:
                    child_list += sudoku2.whatis_rowlist(i, 0)
                    
            children_sudoku_list.append(SudokuObject(child_list))
        
        return children_sudoku_list




testdoku = [0, 8, 0, 0, 0, 0, 0, 9, 0, 
            0, 0, 7, 5, 0, 2, 8, 0, 0,
            6, 0, 0, 8, 0, 7, 0, 0, 5,
            3, 7, 0, 0, 8, 0, 0, 5, 1,
            2, 0, 0, 0, 0, 0, 0, 0, 8,
            9, 5, 0, 0, 4, 0, 0, 3, 2,
            8, 0, 0, 1, 0, 4, 0, 0, 9,
            0, 0, 1, 9, 0, 3, 6, 0, 0,
            0, 4, 0, 0, 0, 0, 0, 2, 0]

s = SudokuObject(testdoku)
s.display()

sga = SudokuGA(s)
sga.run(info=True)

