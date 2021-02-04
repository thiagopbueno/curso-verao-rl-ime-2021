import gym
from collections import defaultdict
from textwrap import indent
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate


class ShoppingCartWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_obs = None
        self._next_obs = None
        self._last_action = None

        self._min_order_number = None
        self._department_freq = defaultdict(int)
        self._aisle_freq = defaultdict(int)
        self._time_freq = defaultdict(int)

        self._common_products: np.ndarray = self.env.data._common_products()
        products_df = self.read_products(self._common_products)
        self.product_names = products_df["product_name"].to_numpy().astype(np.str)

        self.department_ids = products_df["department_id"].to_numpy().astype(np.uint8)
        self.aisle_ids = products_df["aisle_id"].to_numpy().astype(np.uint8)

        self.departments = (
            self.read_departments(self.department_ids)["department"]
            .to_numpy()
            .astype(np.str)
        )
        self.aisles = (
            self.read_aisles(self.aisle_ids)["aisle"].to_numpy().astype(np.str)
        )

        self.dow = np.array("Sun Mon Tue Wed Thu Fri Sat".split(), dtype=np.str)
        self.hod = np.array(
            [f"{h:2d}:00" for h in range(0, 24)],
            dtype=np.str,
        )

    def read_products(self, product_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        dataframe = pd.read_csv(
            self.data.directory / "products.csv",
            dtype={
                "aisle_id": np.uint8,
                "department_id": np.uint8,
                "product_id": np.uint16,
                "product_name": np.str,
            },
            index_col="product_id",
        )

        if product_ids is not None:
            dataframe = dataframe.loc[product_ids]

        return dataframe

    def read_departments(
        self, department_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        dataframe = pd.read_csv(
            self.data.directory / "departments.csv",
            dtype={"department_id": np.uint8, "department": np.str},
            index_col="department_id",
        )

        if department_ids is not None:
            dataframe = dataframe.loc[department_ids]

        return dataframe

    def read_aisles(self, aisle_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        dataframe = pd.read_csv(
            self.data.directory / "aisles.csv",
            dtype={"aisle_id": np.uint8, "aisle": np.str},
            index_col="aisle_id",
        )

        if aisle_ids is not None:
            dataframe = dataframe.loc[aisle_ids]

        return dataframe

    def reset(self) -> np.ndarray:
        self._last_obs = self._next_obs = super().reset()
        self._last_action = None
        self._min_order_number = self.env._user_data.index.min()

        self._department_freq.clear()
        self._aisle_freq.clear()
        self._time_freq.clear()

        return self._next_obs

    def last_order_number(self) -> int:
        return self.env._order_number - 1

    def next_order_number(self) -> int:
        return self.env._order_number
    
    def last_product_ids(self) -> np.ndarray:
        return (
            self.env._purchase_data.loc[self.last_order_number()]
            .to_numpy()
            .astype(np.bool)
        )
        

    def last_orders(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        products = self.last_product_ids()
        return (
            self.product_names[products],
            self.departments[products],
            self.aisles[products],
        )

    def last_order_dow(self) -> str:
        return self.dow[self._last_obs[0:7].argmax()]

    def last_order_hod(self) -> str:
        return self.hod[self._last_obs[7:31].argmax()]

    def days_since_prior_order(self) -> float:
        return self._last_obs[-1]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self._last_obs = self._next_obs
        self._last_action = action
        next_obs, rew, done, info = super().step(action)
        self._next_obs = next_obs

        self.update_freqs()
        info.update(departments=self._department_freq.copy())
        info.update(aisles=self._aisle_freq.copy())
        info.update(times=self._time_freq.copy())

        return next_obs, rew, done, info

    def update_freqs(self):
        _, departments, aisles = self.last_orders()
        for dep in departments:
            self._department_freq[dep] += 1
        for aisle in aisles:
            self._aisle_freq[aisle] += 1
            
        dow, hod = self.last_order_dow(), self.last_order_hod()
        self._time_freq[dow] += 1
        self._time_freq[hod] += 1
        self._time_freq[dow, hod] += 1

    def render(self, mode: str = "human"):
        del mode
        if self._last_action is None:
            return

        action = self._last_action
        visit_number = self.last_order_number() - self._min_order_number

        print(
            f"Visit number {int(visit_number)}, {self.last_order_dow()}, {self.last_order_hod()}"
        )
        print(f"Days since prior order: {self.days_since_prior_order():.2f}")
        
        dataframe = pd.DataFrame(
            {
                "product": self.product_names,
                "department": self.departments,
                "aisle": self.aisles,
                "user": self.last_product_ids(),
                "agent": action.astype(np.bool), 
            }
        ).set_index("product")
        
        dataframe = dataframe[dataframe["user"] | dataframe["agent"]]
        dataframe["user"] = dataframe["user"].astype(np.uint8)
        dataframe["agent"] = dataframe["agent"].astype(np.uint8)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(tabulate(dataframe, headers="keys", tablefmt="psql"))
