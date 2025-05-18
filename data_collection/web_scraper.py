import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='scraper.log'
)
logger = logging.getLogger('ipl_scraper')

class IPLScraper:
    """
    Web scraper for collecting IPL cricket data from various sources.
    """
    
    def __init__(self, sleep_time_range=(1, 3)):
        """
        Initialize the IPL data scraper.
        
        Args:
            sleep_time_range (tuple): Range of seconds to sleep between requests
        """
        self.sleep_time_range = sleep_time_range
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
    
    def _get_random_user_agent(self):
        """Get a random user agent from the list."""
        return random.choice(self.user_agents)
    
    def _make_request(self, url):
        """
        Make an HTTP request with randomized user agent and sleep time.
        
        Args:
            url (str): URL to request
            
        Returns:
            requests.Response: Response object
        
        Raises:
            Exception: If request fails
        """
        headers = {'User-Agent': self._get_random_user_agent()}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Sleep to avoid overloading the server
            time.sleep(random.uniform(*self.sleep_time_range))
            
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for URL {url}: {str(e)}")
            raise
    
    def scrape_match_data(self, url):
        """
        Scrape match data from a specific IPL match page.
        
        Args:
            url (str): URL of the match page
            
        Returns:
            dict: Dictionary containing match information
        """
        logger.info(f"Scraping match data from {url}")
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract match information
            # This is a simplified version - in a real implementation, this would
            # include more sophisticated parsing based on the actual website structure
            
            match_info = {}
            
            # Example extraction (would need to be adjusted for the actual website):
            match_header = soup.find('div', class_='match-header')
            if match_header:
                # Extract teams
                teams = match_header.find_all('p', class_='team-name')
                if len(teams) >= 2:
                    match_info['team1'] = teams[0].text.strip()
                    match_info['team2'] = teams[1].text.strip()
                
                # Extract venue
                venue_elem = match_header.find('p', class_='venue')
                if venue_elem:
                    match_info['venue'] = venue_elem.text.strip()
                
                # Extract date
                date_elem = match_header.find('p', class_='date')
                if date_elem:
                    match_info['date'] = date_elem.text.strip()
            
            # Extract scorecard
            scorecard = soup.find('div', class_='scorecard')
            if scorecard:
                # Extract scores
                team1_score = scorecard.find('div', class_='team1-score')
                team2_score = scorecard.find('div', class_='team2-score')
                
                if team1_score:
                    match_info['team1_score'] = team1_score.text.strip()
                
                if team2_score:
                    match_info['team2_score'] = team2_score.text.strip()
                
                # Extract winner
                result = scorecard.find('p', class_='match-result')
                if result:
                    match_info['result'] = result.text.strip()
                    
                    # Parse winner from result
                    if 'won by' in match_info['result']:
                        winner = match_info['result'].split('won by')[0].strip()
                        match_info['winner'] = winner
            
            # Extract player performances
            player_tables = soup.find_all('table', class_='player-performance')
            
            batting_performances = []
            bowling_performances = []
            
            for table in player_tables:
                table_header = table.find('th')
                if table_header and 'batting' in table_header.text.lower():
                    # Parse batting table
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cols = row.find_all('td')
                        if len(cols) >= 6:
                            batting_performances.append({
                                'player': cols[0].text.strip(),
                                'runs': cols[1].text.strip(),
                                'balls': cols[2].text.strip(),
                                'fours': cols[3].text.strip(),
                                'sixes': cols[4].text.strip(),
                                'strike_rate': cols[5].text.strip()
                            })
                
                elif table_header and 'bowling' in table_header.text.lower():
                    # Parse bowling table
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cols = row.find_all('td')
                        if len(cols) >= 6:
                            bowling_performances.append({
                                'player': cols[0].text.strip(),
                                'overs': cols[1].text.strip(),
                                'maidens': cols[2].text.strip(),
                                'runs': cols[3].text.strip(),
                                'wickets': cols[4].text.strip(),
                                'economy': cols[5].text.strip()
                            })
            
            match_info['batting_performances'] = batting_performances
            match_info['bowling_performances'] = bowling_performances
            
            logger.info(f"Successfully scraped match data")
            return match_info
            
        except Exception as e:
            logger.error(f"Error scraping match data: {str(e)}")
            return None
    
    def scrape_player_stats(self, url):
        """
        Scrape player statistics from an IPL stats page.
        
        Args:
            url (str): URL of the player stats page
            
        Returns:
            pd.DataFrame: DataFrame containing player statistics
        """
        logger.info(f"Scraping player stats from {url}")
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the player stats table
            table = soup.find('table', class_='stats-table')
            
            if not table:
                logger.warning("Player stats table not found")
                return pd.DataFrame()
            
            # Extract table headers
            headers = []
            for th in table.find('thead').find_all('th'):
                headers.append(th.text.strip())
            
            # Extract player data
            player_data = []
            for row in table.find('tbody').find_all('tr'):
                player_row = {}
                for i, td in enumerate(row.find_all('td')):
                    if i < len(headers):
                        player_row[headers[i]] = td.text.strip()
                player_data.append(player_row)
            
            # Convert to DataFrame
            player_df = pd.DataFrame(player_data)
            
            logger.info(f"Successfully scraped player stats, found {len(player_df)} players")
            return player_df
            
        except Exception as e:
            logger.error(f"Error scraping player stats: {str(e)}")
            return pd.DataFrame()
    
    def scrape_team_stats(self, url):
        """
        Scrape team statistics from an IPL team stats page.
        
        Args:
            url (str): URL of the team stats page
            
        Returns:
            pd.DataFrame: DataFrame containing team statistics
        """
        logger.info(f"Scraping team stats from {url}")
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the team stats table
            table = soup.find('table', class_='team-stats')
            
            if not table:
                logger.warning("Team stats table not found")
                return pd.DataFrame()
            
            # Extract table headers
            headers = []
            for th in table.find('thead').find_all('th'):
                headers.append(th.text.strip())
            
            # Extract team data
            team_data = []
            for row in table.find('tbody').find_all('tr'):
                team_row = {}
                for i, td in enumerate(row.find_all('td')):
                    if i < len(headers):
                        team_row[headers[i]] = td.text.strip()
                team_data.append(team_row)
            
            # Convert to DataFrame
            team_df = pd.DataFrame(team_data)
            
            logger.info(f"Successfully scraped team stats, found {len(team_df)} teams")
            return team_df
            
        except Exception as e:
            logger.error(f"Error scraping team stats: {str(e)}")
            return pd.DataFrame()
    
    def scrape_match_schedule(self, url):
        """
        Scrape upcoming match schedule from IPL website.
        
        Args:
            url (str): URL of the match schedule page
            
        Returns:
            pd.DataFrame: DataFrame containing match schedule
        """
        logger.info(f"Scraping match schedule from {url}")
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the schedule table
            table = soup.find('table', class_='schedule')
            
            if not table:
                logger.warning("Schedule table not found")
                return pd.DataFrame()
            
            # Extract match schedule data
            schedule_data = []
            
            for match_div in soup.find_all('div', class_='match-fixture'):
                match = {}
                
                # Extract date
                date_elem = match_div.find('div', class_='fixture-date')
                if date_elem:
                    match['date'] = date_elem.text.strip()
                
                # Extract teams
                teams = match_div.find_all('span', class_='team-name')
                if len(teams) >= 2:
                    match['team1'] = teams[0].text.strip()
                    match['team2'] = teams[1].text.strip()
                
                # Extract venue
                venue_elem = match_div.find('div', class_='venue')
                if venue_elem:
                    match['venue'] = venue_elem.text.strip()
                
                # Extract time
                time_elem = match_div.find('div', class_='match-time')
                if time_elem:
                    match['time'] = time_elem.text.strip()
                
                schedule_data.append(match)
            
            # Convert to DataFrame
            schedule_df = pd.DataFrame(schedule_data)
            
            logger.info(f"Successfully scraped match schedule, found {len(schedule_df)} upcoming matches")
            return schedule_df
            
        except Exception as e:
            logger.error(f"Error scraping match schedule: {str(e)}")
            return pd.DataFrame()

    def scrape_player_recent_form(self, player_name, url):
        """
        Scrape a player's recent form from their profile page.
        
        Args:
            player_name (str): Name of the player
            url (str): URL of the player's profile page
            
        Returns:
            pd.DataFrame: DataFrame containing player's recent performance
        """
        logger.info(f"Scraping recent form for {player_name} from {url}")
        
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the recent matches section
            recent_matches_div = soup.find('div', class_='recent-matches')
            
            if not recent_matches_div:
                logger.warning(f"Recent matches section not found for {player_name}")
                return pd.DataFrame()
            
            # Find the performance table
            table = recent_matches_div.find('table')
            
            if not table:
                logger.warning(f"Recent performance table not found for {player_name}")
                return pd.DataFrame()
            
            # Extract table headers
            headers = []
            for th in table.find('thead').find_all('th'):
                headers.append(th.text.strip())
            
            # Extract recent form data
            form_data = []
            for row in table.find('tbody').find_all('tr'):
                form_row = {}
                for i, td in enumerate(row.find_all('td')):
                    if i < len(headers):
                        form_row[headers[i]] = td.text.strip()
                form_data.append(form_row)
            
            # Convert to DataFrame
            form_df = pd.DataFrame(form_data)
            
            logger.info(f"Successfully scraped recent form for {player_name}, found {len(form_df)} recent matches")
            return form_df
            
        except Exception as e:
            logger.error(f"Error scraping recent form for {player_name}: {str(e)}")
            return pd.DataFrame()
