from sorted_heap import SortedHeap
from analytical_model import AnalyticalModel
import multiprocessing as mp
import random
import copy


class GeneticAlgorithm:

    def __init__(self, data, response, attributes, seed=42, evaluation="rmse", vb_replicate=False,
                 generations=1000, population_size=100, mutate_percent=0.2, parent_percent=0.2,
                 no_change_threshold=20, model_config=None, parallel=False
                 ):
        self.data = data
        self.response = response
        self.attributes_list = attributes
        self.evaluation = evaluation
        self.vb_replicate = vb_replicate
        self.max_attribute_n = int(self.data.shape[0]/10) \
            if len(self.attributes_list) > int(self.data.shape[0]/10) else len(self.attributes_list)
        self.p_generation = []
        self.i_generation = []
        self.model_config = model_config
        self.random = random.Random()
        self.random.seed(a=seed)
        self.best_models = SortedHeap(n=10, target=0.0)
        self.generations = generations
        self.population_n = population_size
        self.mutate_precent = mutate_percent
        self.parent_n = int(population_size * parent_percent)
        self.nochange_threshold = no_change_threshold
        self.parallel = parallel
        self.initialize_population()

    def initialize_population(self):
        for i in range(0, self.population_n):
            attributes = copy.copy(self.attributes_list)
            ri = self.random.randint(2, self.max_attribute_n)
            m = self.random.sample(attributes, ri)
            self.i_generation.append(m)

    def fitness_function(self, data, response, response_label, model_config, one_out):
        m_data = copy.copy(data)
        m_data[response_label] = response
        m = AnalyticalModel(m_data, response_label, one_out=one_out, model_config=model_config)
        return m

    def perform_crossover(self, top_models):
        children = []
        for i in range(0, len(top_models)-1, 2):
            m1 = top_models[i]
            m2 = top_models[i+1]
            m1m2_1 = list(set(m1) | set(m2))
            m1m2_2 = copy.copy(m1m2_1)
            # Child attribute list could include one less or one more attribute than min and max of parents
            # (within the bounds of 2 and attribute n)
            c_min = min(len(m1), len(m2)) - 1
            c_min = c_min if c_min > 2 else 2
            c_max = max(len(m1), len(m2)) + 1
            c_max = c_max if c_max < len(m1m2_1) else len(m1m2_1)
            c1 = self.random.sample(m1m2_1, self.random.randint(c_min, c_max))
            children.append(c1)
            c2 = self.random.sample(m1m2_2, self.random.randint(c_min, c_max))
            children.append(c2)
        return children

    def perform_mutations(self, model_list, add_percent=0.2, delete_percent=0.2):
        mutate_models_i = self.random.sample(range(0, len(model_list)-1), int(len(model_list) * self.mutate_precent))
        for i in mutate_models_i:
            mm = model_list[i]
            add_delete = self.random.random()
            add = False
            delete = False
            flip = False
            if 2 < len(mm) < self.max_attribute_n:
                if add_percent < add_delete < (1-delete_percent):
                    # flip attribute
                    flip = True
                elif add_percent > add_delete:
                    # delete attribute
                    delete = True
                else:
                    # add attribute
                    add = True
            elif len(mm) == 2:
                if add_delete < (1-delete_percent):
                    # flip attribute
                    flip = True
                else:
                    # add attribute
                    add = True
            elif len(mm) == self.max_attribute_n:
                if add_delete > add_percent:
                    # flip attribute
                    flip = True
                else:
                    # delete attribute
                    delete = True
            attribute_dif = list(set(self.attributes_list) - set(mm))
            if len(attribute_dif) == 0:
                flip = False
                add = False
                delete = True
            mutate_i = self.random.randint(0, len(mm)-1)
            if flip:
                mm[mutate_i] = self.random.choice(attribute_dif)
            elif add:
                mm.append(self.random.choice(attribute_dif))
            else:
                del mm[mutate_i]
            model_list[i] = mm
        return model_list

    def execute(self, add_percent=0.2, delete_percent=0.2):
        g = 0
        nochange_n = 0
        generation_sorting = SortedHeap(n=self.population_n, target=0.0)
        stopping_condition = False
        current_min = float("inf")
        if self.parallel:
            concurrent_ga = mp.cpu_count()
            pool = mp.Pool(concurrent_ga)
        while g < self.generations and not stopping_condition:
            # calculate fitness for i_generation
            if self.parallel:
                pool_results = [pool.apply_async(self.fitness_function, (self.data[list(m)], self.data[self.response].values, self.response, self.model_config, self.vb_replicate)) for m in self.i_generation]
                parallel_results = [p.get() for p in pool_results]
                for m in parallel_results:
                    metric = m.evaluate(use=self.evaluation, check_VIF=True, exclude=False)
                    generation_sorting.add(m, metric)
                    self.best_models.add(m, metric)
            else:
                for m in self.i_generation:
                    i_m = self.fitness_function(self.data[list(m)], self.data[self.response].values, self.response, self.model_config, self.vb_replicate)
                    metric = i_m.evaluate(use=self.evaluation, check_VIF=True, exclude=False)
                    generation_sorting.add(i_m, metric)
                    self.best_models.add(i_m, metric)
            top_models = generation_sorting.get_top(self.parent_n)
            crossover_results = self.perform_crossover(top_models)
            death_pool = generation_sorting.get_bottom(self.population_n - 10)  # save top 10 parents
            survivors_n = self.population_n - 30 if len(death_pool) == self.population_n - 10 else len(death_pool)
            survivors = self.random.sample(death_pool, survivors_n)

            new_generation = top_models[:10] + crossover_results + survivors
            new_generation = self.perform_mutations(new_generation, add_percent=add_percent, delete_percent=delete_percent)
            self.i_generation = new_generation

            # Stopping Condition
            print("Generation: {}, fitness distance: {}".format(g, self.best_models.min))
            if current_min > self.best_models.min:
                current_min = self.best_models.min
                nochange_n = 0
            elif current_min == self.best_models.min:
                nochange_n += 1
            if nochange_n >= self.nochange_threshold:
                stopping_condition = True

            generation_sorting.purge()
            g += 1
        if self.parallel:
            pool.close()
            pool.join()
