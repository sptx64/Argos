import pandas as pd
import numpy as np

def get_tickers_path() :
    tickers_sp500_path = 'tickers/tickers_sp500.csv'
    tickers_etoro_path = 'tickers/tickers_etoro.csv'
    tickers_binance_path = 'tickers/tickers_binance.csv'
    return tickers_sp500_path, tickers_etoro_path, tickers_binance_path

def get_dataset_path(market) :
    dataset_path = ''
    if market == 'sp500' :
        dataset_path = "dataset/sp500.sqlite"
    elif market == 'crypto' :
        dataset_path = "dataset/crypto.sqlite"
    return dataset_path

def get_sqlite_word() :
    sqlite_word_list = ['ABORT','ACTION','ADD','AFTER','ALL','ALTER','ALWAYS','ANALYZE','AND','AS','ASC','ATTACH','AUTOINCREMENT','BEFORE',
                    'BEGIN','BETWEEN','BY','CASCADE','CASE','CAST','CHECK','COLLATE','COLUMN','COMMIT','CONFLICT','CONSTRAINT','CREATE',
                    'CROSS','CURRENT','CURRENT_DATE','CURRENT_TIME','CURRENT_TIMESTAMP','DATABASE','DEFAULT','DEFERRABLE','DEFERRED',
                    'DELETE','DESC','DETACH','DISTINCT','DO','DROP','EACH','ELSE','END','ESCAPE','EXCEPT','EXCLUDE','EXCLUSIVE','EXISTS',
                    'EXPLAIN','FAIL','FILTER','FIRST','FOLLOWING','FOR','FOREIGN','FROM','FULL','GENERATED','GLOB','GROUP','GROUPS',
                    'HAVING','IF','IGNORE','IMMEDIATE','IN','INDEX','INDEXED','INITIALLY','INNER','INSERT','INSTEAD','INTERSECT','INTO',
                    'IS','ISNULL','JOIN','KEY','LAST','LEFT','LIKE','LIMIT','MATCH','MATERIALIZED','NATURAL','NO','NOT','NOTHING','NOTNULL',
                    'NULL','NULLS','OF','OFFSET','ON','OR','ORDER','OTHERS','OUTER','OVER','PARTITION','PLAN','PRAGMA','PRECEDING','PRIMARY',
                    'QUERY','RAISE','RANGE','RECURSIVE','REFERENCES','REGEXP','REINDEX','RELEASE','RENAME','REPLACE','RESTRICT','RETURNING',
                    'RIGHT','ROLLBACK','ROW','ROWS','SAVEPOINT','SELECT','SET','TABLE','TEMP','TEMPORARY','THEN','TIES','TO','TRANSACTION',
                    'TRIGGER','UNBOUNDED','UNION','UNIQUE','UPDATE','USING','VACUUM','VALUES','VIEW','VIRTUAL','WHEN','WHERE','WINDOW',
                    'WITH','WITHOUT']
    return sqlite_word_list
