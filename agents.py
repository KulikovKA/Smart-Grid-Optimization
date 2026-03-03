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
    def __init__(self, agent_id, household_type='residential'):
        super().__init__(agent_id, household_type)
        
        unique_seed = abs(hash(agent_id)) % (2**32)
        self.rng = np.random.RandomState(unique_seed)
        
        self.charge_p = self.rng.randint(5, 25)  
        self.discharge_ratio = self.rng.uniform(0.85, 0.98)
        self.min_spread = self.rng.uniform(3.0, 5.0)

        caps = {'residential': 15000, 'commercial': 200000, 'industrial': 2000000}
        self.battery = Battery(capacity=caps.get(household_type, 10000))
        self.max_charge_speed = self.battery.capacity / 6.0 

    def reset(self):
        """Сбрасывает состояние агента перед тестом новой модели."""
        if self.battery:
            self.battery.current_charge = self.battery.capacity * 0.5

    def optimize_step(self, hour, day_type, season, price_forecast_24h, real_price_now):
        # --- 1. Load Shifting ---
        best_hours_idx = np.argsort(price_forecast_24h[:12])[:3]
        best_hour_idx = self.rng.choice(best_hours_idx)
        best_hour = (hour + best_hour_idx) % 24
        
        avg_price = np.mean(price_forecast_24h)
        
        if real_price_now > avg_price:
            for appliance in self.appliances:
                if isinstance(appliance, ScheduledAppliance) and appliance.can_shift:
                    appliance.try_shift(best_hour)

        base_load = self.measure_consumption(hour, day_type, season)
        if self.battery is None: return base_load

        # --- 2. Battery Logic ---
        current_pred_price = price_forecast_24h[0]
        future_prices = price_forecast_24h[1:]
        
        if len(future_prices) == 0: return base_load
        
        pred_max = np.max(future_prices)
        pred_min = np.min(future_prices)
        
        is_peak_price = (current_pred_price >= pred_max * self.discharge_ratio)
        is_profitable_sell = (real_price_now - pred_min > self.min_spread)
        
        if is_peak_price and is_profitable_sell:
            discharge_amount = min(self.battery.current_charge, self.battery.capacity * 0.2)
            return base_load - self.battery.discharge(discharge_amount)

        charge_threshold = np.percentile(future_prices, self.charge_p)
        is_profitable_buy = (pred_max - current_pred_price > self.min_spread)
        
        if current_pred_price <= charge_threshold and is_profitable_buy:
            if self.battery.current_charge < self.battery.capacity:
                space_left = self.battery.capacity - self.battery.current_charge
                charge_amount = min(space_left, self.max_charge_speed)
                return base_load + self.battery.charge(charge_amount)

        return base_load


class SmartGridAgent:
    """Главный агент умной электросети"""
    def __init__(self, num_residential=60, num_commercial=80, num_industrial=1):
        self.agents = []
        
        for i in range(num_residential):
            self.agents.append(SmartHouseholdAgent(f'res_{i}', 'residential'))
            
        for i in range(num_commercial):
            self.agents.append(SmartHouseholdAgent(f'com_{i}', 'commercial'))
            
        for i in range(num_industrial):
            self.agents.append(SmartHouseholdAgent(f'ind_{i}', 'industrial'))
        
        self.total_consumption_history = []
        self.timestamp_history = []
        self.prev_noise = 0
    
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