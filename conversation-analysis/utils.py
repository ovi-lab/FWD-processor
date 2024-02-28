"""Module with helper functions for JSON-processor to use"""

def link_exists(link_id, link_section):
        """
        Iterate through the global variable mapped_conversation
        and return true if a link with the desired link_id is found,
        else false.

        Parameters
        ----------
        link_id : int
            id of link to find

        Returns
        -------
        boolean
            true if link is found
        """
        for l in link_section:
            if l["name"] == link_id:
                return True
        return False

def node_has_links(n, link_section):
        """
        Iterate through the global variable mapped_conversation
        and return true if a link contains the specified node,
        else false.

        Parameters
        ----------
        n : Object
            A node object with integer attribute 'id'

        Returns
        -------
        boolean
            true if link contains node n
        """
        for l in link_section:
            
            if n["id"] in [l["source"], l["target"]]:
                return True
        return False