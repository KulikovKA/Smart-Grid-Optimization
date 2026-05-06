import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class ElectricAppliance(ABC):
    """Абстрактный класс электроприбора"""
    def __init__(self, name, nominal_power, t0, t1):
        self.name = name
        self.nominal_power = nominal_power
        self.t0 = t0
        self.t1 = t1
    
    @abstractmethod
    def measure_consumption(self, hour, day_type, season):
        """Расчет потребления прибора"""
        pass


class EssentialAppliance(ElectricAppliance):
    """Необходимые приборы (холодильник, роутер) - всегда работают"""
    def __init__(self, name, nominal_power):
        super().__init__(name, nominal_power, 0, 24)
    
    def measure_consumption(self, hour, day_type, season):
        return self.nominal_power


class ScheduledAppliance(ElectricAppliance):
    """Приборы по расписанию (свет, чайник, стиралка) с возможностью сдвига"""
    def __init__(self, name, nominal_power, t0, t1, can_shift=False):
        super().__init__(name, nominal_power, t0, t1)
        self.can_shift = can_shift
        self.shifted_t0 = t0       # Инициализируем текущим временем старта
        self.duration = (t1 - t0) % 24 # Запоминаем длительность работы

    def measure_consumption(self, hour, day_type, season):
        # 1. Определяем базовое время старта с учетом праздников
        is_holiday = day_type in ['weekend', 'holiday_day_1_2', 'holiday_day_3_8',
                                  'may_holiday_1_4', 'may_working_week', 'may_holiday_8_11',
                                  'other_holiday', 'shortened_day']
        
        if is_holiday:
            holiday_t0 = (self.t0 + 1.5) % 24
        else:
            holiday_t0 = self.t0
            
        # 2. Определяем ФАКТИЧЕСКОЕ время старта
        if self.can_shift and self.shifted_t0 != self.t0:
             effective_t0 = self.shifted_t0
        else:
             effective_t0 = holiday_t0
            
        # 3. Считаем время окончания на основе длительности
        effective_t1 = (effective_t0 + self.duration) % 24
        
        # 4. Проверяем, работает ли прибор в текущий час
        if effective_t0 <= effective_t1:
            if effective_t0 <= hour < effective_t1:
                return self.nominal_power
        else: # Переход через полночь (например, старт в 23:00, конец в 01:00)
            if hour >= effective_t0 or hour < effective_t1:
                return self.nominal_power
                
        return 0.0

    def try_shift(self, new_start_hour):
        """Сдвигает время запуска, если это разрешено"""
        if self.can_shift:
            # Ограничение: нельзя двигать больше чем на 6 часов
            if abs(new_start_hour - self.t0) <= 6:
                self.shifted_t0 = new_start_hour
                return True
        return False


class PeakAppliance(ElectricAppliance):
    """Приборы с пиковым потреблением"""
    def __init__(self, name, nominal_power, peak_hours_start, peak_hours_end):
        super().__init__(name, nominal_power, peak_hours_start, peak_hours_end)
    
    def measure_consumption(self, hour, day_type, season):
        if hour < self.t0 or hour > self.t1:
            return 0.0
        
        peak_hour = (self.t0 + self.t1) / 2
        
        if hour <= peak_hour:
            ratio = (hour - self.t0) / (peak_hour - self.t0)
        else:
            ratio = (self.t1 - hour) / (self.t1 - peak_hour)
        
        return self.nominal_power * max(0, ratio)


class TemperatureSensitiveAppliance(ElectricAppliance):
    """Приборы, зависящие от сезона (отопление, кондиционер)"""
    def __init__(self, name, nominal_power, appliance_type='heating'):
        super().__init__(name, nominal_power, 0, 24)
        self.appliance_type = appliance_type
    
    def measure_consumption(self, hour, day_type, season):
        if self.appliance_type == 'heating':
            seasonal_multipliers = {
                'winter': 1.15, 'spring': 0.70, 'summer': 0.50, 'autumn': 0.85
            }
        else: # cooling
            seasonal_multipliers = {
                'winter': 0.20, 'spring': 0.50, 'summer': 1.00, 'autumn': 0.60
            }
        return self.nominal_power * seasonal_multipliers.get(season, 1.0)


class Battery:
    def __init__(self, capacity, efficiency=0.9, max_power=None):
        self.capacity = capacity       # Полная емкость
        self.current_charge = 0.0      # Текущий заряд
        self.efficiency = efficiency   
        
        if max_power is None:
            self.max_power = capacity * 0.5
        else:
            self.max_power = max_power

    def charge(self, amount_needed):
        """Попытка зарядить батарею на amount_needed (Вт)."""
        power_limit = min(amount_needed, self.max_power)
        space_left = self.capacity - self.current_charge
        
        actual_in = min(power_limit, space_left)
        self.current_charge += actual_in * self.efficiency
        
        return actual_in 

    def discharge(self, amount_needed):
        """Попытка разрядить батарею для покрытия amount_needed (Вт)."""
        power_limit = min(amount_needed, self.max_power)
        actual_out = min(power_limit, self.current_charge)
        self.current_charge -= actual_out
        return actual_out


class ElectricVehicle:
    """Электромобиль с батареей и расписанием использования (для Казани)
    
    Поддерживаемые модели:
    - Премиум (западные): Tesla Model 3, VW ID.4
    - Средний класс (западные): Hyundai Kona, Nissan Leaf
    - Премиум (китайские): Liaoxiang Li 9, NIO ES6, Li One
    - Средний класс (китайские): XPeng P7, BYD Qin, Geely Geometry, BYD Song
    """
    def __init__(self, vehicle_id, model='Tesla Model 3'):
        self.vehicle_id = vehicle_id
        self.model = model
        
        # Батарея EV (типичные объемы в kWh, переводим в Дж)
        battery_capacities = {
            # Премиум (западные)
            'Tesla Model 3': 75000 * 3600,              # 75 kWh
            'Volkswagen ID.4': 62000 * 3600,            # 62 kWh
            
            # Средний класс (западные)
            'Nissan Leaf': 40000 * 3600,                # 40 kWh
            'Hyundai Kona Electric': 60000 * 3600,      # 60 kWh
            
            # Премиум (китайские)
            'Liaoxiang Li 9': 85000 * 3600,             # 85 kWh (Li 9 мере)
            'NIO ES6': 100000 * 3600,                   # 100 kWh (премиум)
            'Li One': 105000 * 3600,                    # 105 kWh (флагманская)
            
            # Средний класс (китайские)
            'XPeng P7': 70000 * 3600,                   # 70 kWh
            'BYD Qin Plus EV': 52000 * 3600,            # 52 kWh
            'BYD Song Plus DM-i': 55000 * 3600,         # 55 kWh
            'Geely Geometry A': 51000 * 3600,           # 51 kWh
            'Chery EQ5': 48000 * 3600,                  # 48 kWh
        }
        
        self.battery = Battery(
            capacity=battery_capacities.get(model, 75000 * 3600),
            efficiency=0.92,
            max_power=11000  # 11 kW зарядка (однофазная 380В)
        )
        
        # Расписание использования (когда машина в пути)
        self.away_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # В пути 8:00-18:00
        
        # Суточный расход энергии зависит от размера батареи (больше батарея = чаще заряжается)
        # Базовый расход: 15-20 kWh на 100 км, примерно 30 км в день
        battery_kwh = battery_capacities.get(model, 75000) / 3600 / 1000
        self.daily_consumption_wh = 5250 + int((battery_kwh - 75) * 15)  # Корректировка по объему батареи
        
        # Стратегия зарядки
        self.rng = np.random.RandomState(abs(hash(vehicle_id)) % (2**32))
        self.preferred_charge_hour = self.rng.choice([22, 23, 0, 1, 2])  # Ночная зарядка
        
        # Инициализация батареи (70% заряда в начале дня)
        self.battery.current_charge = self.battery.capacity * 0.7
    
    def get_state_of_charge(self):
        """Получить процент заряда батареи (0.0 - 1.0)"""
        return self.battery.current_charge / self.battery.capacity
    
    def simulate_driving(self):
        """Симуляция расхода энергии при движении (8:00-18:00)"""
        # Разряжаем батарею за день в пути (30 км)
        daily_consumption_j = self.daily_consumption_wh * 3600
        return self.battery.discharge(daily_consumption_j)
    
    def get_charging_demand(self, hour, day_type):
        """Спрос на зарядку машины (базовый расчет)"""
        discharge_multiplier = 0.7 if day_type == 'weekend' else 1.0
        daily_need = self.daily_consumption_wh * discharge_multiplier
        return daily_need / 8.0  # Базовая потребность за час

    def smart_charging_decision(self, price_forecast_24h, grid_load_forecast=None):
        """
        Динамическое решение о зарядке/разрядке на основе прогноза цен и нагрузки.
        
        Возвращает: 'fast_charge', 'charge', 'hold', 'discharge'
        """
        avg_price = np.mean(price_forecast_24h)
        current_soc = self.get_state_of_charge()
        current_price = price_forecast_24h[0]
        
        # СЦЕНАРИЙ 1: Цена очень низкая (< 70% от средней) → агрессивная зарядка
        if current_price < avg_price * 0.7 and current_soc < 0.95:
            return 'fast_charge'
        
        # СЦЕНАРИЙ 2: Предстоит пик нагрузки → подготовиться
        if grid_load_forecast is not None and len(grid_load_forecast) > 0:
            avg_load = np.mean(grid_load_forecast)
            next_peak_hours = np.where(grid_load_forecast > avg_load * 1.15)[0]
            
            if len(next_peak_hours) > 0 and min(next_peak_hours) < 4:
                if current_soc < 0.8:
                    return 'charge'
                elif current_soc > 0.5:
                    return 'discharge'  # V2G: помощь сети в пик
        
        # СЦЕНАРИЙ 3: Обычный режим (цена/нагрузка средние)
        if current_price < avg_price and current_soc < 0.8:
            return 'charge'
        elif current_price > avg_price * 1.2 and current_soc > 0.4:
            return 'discharge'  # V2G
        
        return 'hold'


class HouseholdAgent:
    """Базовый агент домохозяйства"""
    def __init__(self, agent_id, household_type='residential'):
        self.agent_id = agent_id
        self.household_type = household_type
        self.appliances = []
        self.current_hour_consumption = 0.0
        self._initialize_appliances()
    
    def _initialize_appliances(self):
            if self.household_type == 'residential':
                self.appliances.append(EssentialAppliance("Холодильник", 180))       
                self.appliances.append(EssentialAppliance("Роутер+УмныйДом", 30))
                self.appliances.append(EssentialAppliance("Фоновые устройства", 50)) 
                self.appliances.append(ScheduledAppliance("Свет_Утро", 150, 6, 9))   
                self.appliances.append(ScheduledAppliance("Свет_Вечер", 250, 17, 23)) 
                self.appliances.append(ScheduledAppliance("Чайник", 2000, 7, 8))     
                self.appliances.append(ScheduledAppliance("Микроволновка", 1200, 18, 19)) 
                self.appliances.append(PeakAppliance("Электроплита", 3500, 18, 20))  
                self.appliances.append(ScheduledAppliance("Стиральная машина", 2200, 19, 21, can_shift=True))
                self.appliances.append(ScheduledAppliance("Посудомоечная машина", 1500, 21, 23, can_shift=True))
                self.appliances.append(ScheduledAppliance("Зарядка EV (Tesla)", 7000, 19, 23, can_shift=True))
                self.appliances.append(TemperatureSensitiveAppliance("Теплый пол/Отопление", 1500, 'heating'))
                self.appliances.append(TemperatureSensitiveAppliance("Кондиционер", 1200, 'cooling'))
                self.appliances.append(PeakAppliance("Водонагреватель (Душ)", 3000, 7, 8)) 
                self.appliances.append(ScheduledAppliance("Игровой ПК / ТВ", 300, 19, 23))

            elif self.household_type == 'commercial':
                self.appliances.append(EssentialAppliance("Серверная + Сеть", 2500)) 
                self.appliances.append(EssentialAppliance("Базовое обеспечение", 1500))
                self.appliances.append(ScheduledAppliance("Освещение (Open Space)", 4000, 8, 19))
                self.appliances.append(ScheduledAppliance("Офисная техника", 12500, 9, 18))
                self.appliances.append(ScheduledAppliance("Кухня офиса", 3000, 12, 14))
                self.appliances.append(TemperatureSensitiveAppliance("HVAC Система", 20000, 'cooling')) 
                self.appliances.append(TemperatureSensitiveAppliance("Тепловая завеса", 15000, 'heating'))
                self.appliances.append(PeakAppliance("Лифтовая группа", 8000, 8, 19))

            elif self.household_type == 'industrial':
                self.appliances.append(EssentialAppliance("Системы жизнеобеспечения", 30000)) 
                self.appliances.append(ScheduledAppliance("Станочный парк", 150000, 7, 20))
                self.appliances.append(ScheduledAppliance("Конвейерная линия", 80000, 7, 20))
                self.appliances.append(ScheduledAppliance("Термообработка", 100000, 8, 18, can_shift=True))
                self.appliances.append(ScheduledAppliance("Промышленный свет", 15000, 6, 22))
                self.appliances.append(PeakAppliance("Компрессорная станция", 40000, 8, 17))
                self.appliances.append(TemperatureSensitiveAppliance("Пром. вентиляция", 50000, 'cooling'))
                self.appliances.append(TemperatureSensitiveAppliance("Пром. отопление", 60000, 'heating'))

    def measure_consumption(self, hour, day_type, season):
        self.current_hour_consumption = 0.0
        for appliance in self.appliances:
            self.current_hour_consumption += appliance.measure_consumption(hour, day_type, season)
        return self.current_hour_consumption


class SmartHouseholdAgent(HouseholdAgent):
    def __init__(self, agent_id, household_type='residential', has_ev=False, ev_model='Tesla Model 3'):
        super().__init__(agent_id, household_type)
        
        unique_seed = abs(hash(agent_id)) % (2**32)
        self.rng = np.random.RandomState(unique_seed)
        
        self.charge_p = self.rng.randint(5, 25)  
        self.discharge_ratio = self.rng.uniform(0.85, 0.98)
        self.min_spread = self.rng.uniform(3.0, 5.0)

        caps = {'residential': 15000, 'commercial': 200000, 'industrial': 2000000}
        self.battery = Battery(capacity=caps.get(household_type, 10000))
        self.max_charge_speed = self.battery.capacity / 6.0 
        
        # Электромобиль (только для резидентциальных домохозяйств с EV)
        self.electric_vehicle = None
        self.ev_fleet = []  # Список всех EV в домохозяйстве
        
        if has_ev and household_type == 'residential':
            # Может быть 1-3 машины в доме (реалистично)
            num_evs = self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            for i in range(num_evs):
                ev = ElectricVehicle(f"{agent_id}_ev{i}", ev_model)
                self.ev_fleet.append(ev)
            self.electric_vehicle = self.ev_fleet[0]  # Для обратной совместимости

    def reset(self):
        """Сбрасывает состояние агента перед тестом новой модели."""
        if self.battery:
            self.battery.current_charge = self.battery.capacity * 0.5
        # Сбросить все EV
        for ev in self.ev_fleet:
            ev.battery.current_charge = ev.battery.capacity * 0.7

    def _distribute_ev_charging(self, available_power, price_forecast_24h, real_price_now, current_hour, grid_load_forecast=None):
        """
        Распределенная динамическая зарядка парка EV на основе цен и прогноза нагрузки.
        Система сама решает когда заряжать, разряжать или ждать.
        """
        total_charged = 0.0
        total_discharged = 0.0
        
        if not self.ev_fleet:
            return 0.0
        
        # Симулируем разрядку при движении (8:00-18:00)
        for ev in self.ev_fleet:
            if 8 <= current_hour <= 18:
                ev.simulate_driving()
        
        # ДИНАМИЧЕСКАЯ ЗАРЯДКА: каждая машина решает сама на основе прогноза
        remaining_power = available_power
        
        for ev in self.ev_fleet:
            if remaining_power <= 0:
                break
            
            # Получить решение машины (на основе цен и нагрузки)
            charging_decision = ev.smart_charging_decision(price_forecast_24h, grid_load_forecast)
            current_soc = ev.get_state_of_charge()
            base_demand = ev.get_charging_demand(current_hour, 'weekday')
            
            # FAST_CHARGE: очень низкие цены (купить максимум дешевой энергии)
            if charging_decision == 'fast_charge':
                charge_amount = min(
                    ev.battery.capacity * 0.4,  # Усиливаем эффект: до 40% в час
                    remaining_power,
                    ev.battery.max_power
                )
                charged = ev.battery.charge(charge_amount)
                total_charged += charged
                remaining_power -= charged
            
            # CHARGE: цена благоприятная (нормальная зарядка)
            elif charging_decision == 'charge':
                charge_amount = min(base_demand * 1.5, remaining_power, ev.battery.max_power)
                charged = ev.battery.charge(charge_amount)
                total_charged += charged
                remaining_power -= charged
            
            # DISCHARGE: V2G (отдаем энергию в сеть, помогаем в пик)
            elif charging_decision == 'discharge' and current_soc > 0.35:
                # Отдаем до 20% батареи сети
                discharge_amount = min(ev.battery.current_charge * 0.2, ev.battery.capacity * 0.35)
                discharged = ev.battery.discharge(discharge_amount)
                total_discharged += discharged
            
            # HOLD: ждем лучших условий (ничего не делаем)
            # else: pass
        
        # Возвращаем NET значение: положительное = зарядили, отрицательное = разрядили
        return total_charged - total_discharged

    def optimize_step(self, hour, day_type, season, price_forecast_24h, real_price_now):
        # --- 1. Load Shifting (сдвиг нагрузки домашних приборов) ---
        best_hours_idx = np.argsort(price_forecast_24h[:12])[:3]
        best_hour_idx = self.rng.choice(best_hours_idx)
        best_hour = (hour + best_hour_idx) % 24
        
        avg_price = np.mean(price_forecast_24h)
        
        if real_price_now > avg_price:
            for appliance in self.appliances:
                if isinstance(appliance, ScheduledAppliance) and appliance.can_shift:
                    appliance.try_shift(best_hour)

        base_load = self.measure_consumption(hour, day_type, season)
        
        # --- 2. EV Fleet Charging Optimization (динамическая зарядка) ---
        ev_load = 0.0
        if self.ev_fleet:
            # Умная динамическая зарядка (система сама решает на основе цен)
            ev_load = self._distribute_ev_charging(
                available_power=self.battery.max_power * 2,
                price_forecast_24h=price_forecast_24h,
                real_price_now=real_price_now,
                current_hour=hour,
                grid_load_forecast=None  # Можно передать реальный прогноз нагрузки если есть
            )
        
        if self.battery is None: 
            return base_load + ev_load

        # --- 3. Home Battery Arbitrage (домашняя батарея для арбитража) ---
        current_pred_price = price_forecast_24h[0]
        future_prices = price_forecast_24h[1:]
        
        if len(future_prices) == 0: 
            return base_load + ev_load
        
        pred_max = np.max(future_prices)
        pred_min = np.min(future_prices)
        
        is_peak_price = (current_pred_price >= pred_max * self.discharge_ratio)
        is_profitable_sell = (real_price_now - pred_min > self.min_spread)
        
        if is_peak_price and is_profitable_sell:
            discharge_amount = min(self.battery.current_charge, self.battery.capacity * 0.2)
            return base_load + ev_load - self.battery.discharge(discharge_amount)

        charge_threshold = np.percentile(future_prices, self.charge_p)
        is_profitable_buy = (pred_max - current_pred_price > self.min_spread)
        
        if current_pred_price <= charge_threshold and is_profitable_buy:
            if self.battery.current_charge < self.battery.capacity:
                space_left = self.battery.capacity - self.battery.current_charge
                charge_amount = min(space_left, self.max_charge_speed)
                return base_load + ev_load + self.battery.charge(charge_amount)

        return base_load + ev_load


class SmartGridAgent:
    """Главный агент умной электросети (для Казани)
    
    Состав потребителей:
    - 60 жилых домохозяйств (из них 10 имеют электромобили ~17%)
    - 80 коммерческих объектов
    - 1 промышленный объект
    
    Поддерживаемые модели EV (13 моделей):
    - Западные: Tesla Model 3, VW ID.4, Nissan Leaf, Hyundai Kona Electric
    - Китайские премиум: Liaoxiang Li 9, NIO ES6, Li One
    - Китайские средний класс: XPeng P7, BYD Qin Plus, BYD Song Plus, Geely Geometry, Chery EQ5
    
    Итого: 141 потребитель + 10 EV = 151 агент нагрузки
    """
    def __init__(self, num_residential=60, num_commercial=80, num_industrial=1, num_ev_households=20):
        self.agents = []
        self.ev_agents = []
        
        # Жилые домохозяйства (включая те, что с EV)
        ev_models = [
            # Западные
            'Tesla Model 3',
            'Volkswagen ID.4',
            'Nissan Leaf',
            'Hyundai Kona Electric',
            # Китайские премиум
            'Liaoxiang Li 9',
            'NIO ES6',
            'Li One',
            # Китайские средний класс
            'XPeng P7',
            'BYD Qin Plus EV',
            'BYD Song Plus DM-i',
            'Geely Geometry A',
            'Chery EQ5'
        ]
        ev_indices = np.random.choice(num_residential, size=min(num_ev_households, num_residential), replace=False)
        
        for i in range(num_residential):
            has_ev = i in ev_indices
            ev_model = np.random.choice(ev_models) if has_ev else None
            agent = SmartHouseholdAgent(f'res_{i}', 'residential', has_ev=has_ev, ev_model=ev_model)
            self.agents.append(agent)
            if has_ev:
                self.ev_agents.append(agent)
            
        # Коммерческие объекты (офисы, магазины)
        for i in range(num_commercial):
            self.agents.append(SmartHouseholdAgent(f'com_{i}', 'commercial'))
            
        # Промышленные объекты
        for i in range(num_industrial):
            self.agents.append(SmartHouseholdAgent(f'ind_{i}', 'industrial'))
        
        self.total_consumption_history = []
        self.timestamp_history = []
        self.prev_noise = 0
        
        print(f" SmartGrid инициализирована:")
        print(f"  - Жилых домохозяйств: {num_residential} (из них {len(self.ev_agents)} с EV)")
        print(f"  - Коммерческих объектов: {num_commercial}")
        print(f"  - Промышленных объектов: {num_industrial}")
        print(f"  - Всего агентов: {len(self.agents)} + {len(self.ev_agents)} с EV")
        print(f"  - Доступных моделей EV: {len(ev_models)}")
        
        # Статистика распределения моделей
        ev_models_actual = [agent.electric_vehicle.model for agent in self.ev_agents if agent.electric_vehicle]
        if ev_models_actual:
            from collections import Counter
            model_counts = Counter(ev_models_actual)
            print(f"\n  Распределение моделей EV:")
            for model, count in sorted(model_counts.items()):
                print(f"     {model}: {count} шт.")
    
    def step(self, timestamp, day_type, season):
        """Один шаг генерации данных (обучающая выборка)"""
        hour = timestamp.hour
        total = 0.0
        
        for agent in self.agents:
            consumption = agent.measure_consumption(hour, day_type, season)
            total += consumption
        
        holiday_multipliers = {
            'weekday': 1.0, 'weekend': 0.95, 'new_year_eve': 1.0, 
            'new_year_morning': 0.87, 'holiday_day_1_2': 0.87,
            'may_holiday_1_4': 0.945, 'other_holiday': 0.925
        }
        base_multiplier = holiday_multipliers.get(day_type, 1.0)
        
        day_of_year = (timestamp - pd.Timestamp("2025-01-01")).days
        seasonal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * day_of_year / 365.25 + 1.5)
        
        total = total * base_multiplier * seasonal_factor
        
        noise = np.random.normal(0, total * 0.02)
        total += noise
        
        self.total_consumption_history.append(max(0, total))
        self.timestamp_history.append(timestamp)
        
        return max(0, total)

    def get_total_consumption(self):
        return np.array(self.total_consumption_history)